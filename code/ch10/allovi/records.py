"""Decision traces: one row per period and one row per candidate (Section 8.1).

The core records its own quantities: the requirement, the number of
candidates evaluated and admitted, the committed entry, its ambition, level
and onset level, the reason for the choice, and for every candidate its
profile, keys and outcome. What a plan, an initial set, a proposal or a state
is, only the modules know; they describe them through optional diagnostic
methods (describe_plan, describe_set, describe_proposal, describe_state),
whose columns are added to the rows. Diagnostic methods never influence a
decision.
"""
from __future__ import annotations

import csv
from typing import Callable, Iterable, List

from .levels import NEG_INF


def _describe(module, method, *args):
    fn = getattr(module, method, None)
    return dict(fn(*args)) if fn is not None else {}


def decision_rows(record, control, robot) -> List[dict]:
    rows = []
    for p in record.periods:
        req = p.requirement
        row = dict(k=p.k, t=p.t, governor=p.governor, req=req.level, onset=req.onset)
        row.update(_describe(robot, "describe_state", p.measured))
        d = p.decision
        if p.governor == "normal":
            row.update(_describe(control, "describe_set", p.initial_set))
            row.update(_describe(control, "describe_proposal", p.proposal, p.initial_set))
            row.update(candidates=len(d.entries), evaluated=d.evaluated,
                       admitted=len(d.keys), selected=d.selected, ambition=d.ambition,
                       level=d.profile.level, onset_level=d.profile.at(req.onset),
                       reason=d.reason)
        else:
            row.update(_describe(control, "describe_proposal", p.proposal, p.measured))
            if d is None:
                row.update(candidates=0, evaluated=0, admitted=0, selected=None,
                           ambition=0, level=None, onset_level=None, reason=p.reach_reason)
            else:
                committed = d.plan is not None
                row.update(candidates=len(d.entries), evaluated=d.evaluated,
                           admitted=len(d.keys), selected=d.selected,
                           ambition=d.ambition if committed else 0,
                           level=d.profile.level if committed else None,
                           onset_level=d.profile.at(req.onset) if committed else None,
                           reason=d.reason)
        rows.append(row)
    return rows


def candidate_rows(record, control) -> List[dict]:
    rows = []
    for p in record.periods:
        d = p.decision
        if d is None:
            continue
        onset = p.requirement.onset
        for e in d.entries:
            key = d.keys.get(e.index)
            why = e.reasons or []
            row = dict(k=p.k, t=p.t, index=e.index, candidate=e.kind, ambition=e.ambition)
            row.update(_describe(control, "describe_plan", e.plan))
            row.update(level=e.profile.level, onset_level=e.profile.at(onset),
                       level_reason=why[0].text if why else "",
                       onset_reason=why[min(onset, len(why) - 1)].text if why else "",
                       admitted=key is not None,
                       key1=None if key is None else key[0],
                       key2=None if key is None else key[1],
                       outcome=d.outcomes[e.index])
            rows.append(row)
    return rows


def write_csv(path, columns: Iterable[str], rows: Iterable[dict],
              fmt: Callable = str, empty_uncertified=True):
    """Writes rows; None as empty, booleans as yes/no, NEG_INF as empty
    (empty_uncertified) or '-inf', and every other value through fmt."""
    columns = list(columns)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(columns)
        for row in rows:
            out = []
            for c in columns:
                x = row.get(c)
                if x is None:
                    out.append("")
                elif isinstance(x, bool):
                    out.append("yes" if x else "no")
                elif isinstance(x, float) and x == NEG_INF:
                    out.append("" if empty_uncertified else "-inf")
                else:
                    out.append(fmt(x))
            w.writerow(out)
