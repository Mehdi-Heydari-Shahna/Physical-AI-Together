#!/usr/bin/env python3
"""
The path of a contribution, from registration to a report (Section 17.1).

Physical AI, Together, Chapter 17.

For each of the two worked contributions of contrib/, a control package
(reader.full-reverse) and a demand predictor (reader.early-window), this
script
  * registers the module through the public interface of the core
    (../ch10/allovi) and reports its role, name and contract;
  * runs the check suite of Section 10.3 with the module in its role in the
    configuration of the forecasting run of Chapter 3;
  * runs the module in the world of Chapter 3: for the control package, the
    shortest lead time at which the operation begins on time; for the
    demand predictor, the run with the schedule of Chapter 3;
  * compares it with ALLOVI as Chapter 16 configured it (C6), replacing one
    module and nothing else, in a family of 100 worlds that Chapter 16 did
    not use (seeds 16101 to 16200), and applies the decision rules of the
    hypotheses that HYPOTHESES.md registered before the first run;
  * writes the contribution report.

Usage
    python3 contribute.py            # writes results/contribution-*
    python3 contribute.py --check    # also checks every value of the report
    python3 contribute.py --jobs 1   # one process (default: up to two)

Outputs, in results/ next to this file
    contribution-summary.txt   every value of the report; with --check, the
                               result
    contribution-runs.csv      one row per configuration and world

Python 3.8 or later; no other packages. About four minutes with two
processes. The results do not depend on the number of processes.
"""
import argparse
import csv
import os
import sys
from fractions import Fraction as Fr
from multiprocessing import Pool

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
CH16 = os.path.join(HERE, "..", "ch16")
sys.path.insert(0, CH16)
sys.path.insert(0, HERE)
import cart_benchmark as cb                                       # noqa: E402
import allovi                                                     # noqa: E402
import cart                                                       # noqa: E402
from allovi import checks                                         # noqa: E402
from bench import modules as bm                                   # noqa: E402
from bench import stats                                           # noqa: E402
from bench import worlds as bw                                    # noqa: E402
from cart.model import ce                                         # noqa: E402

import contrib.early_window                                       # noqa: E402
import contrib.full_reverse                                       # noqa: E402

FAMILY = range(16101, 16201)
PERIODS = cb.PERIODS
EARLY = 10
KINDS = ("C6", "full-reverse", "early-window")
CONTRIBUTIONS = (("full-reverse", "control", "reader.full-reverse"),
                 ("early-window", "demand", "reader.early-window"))
CLASSES = {"reader.full-reverse": contrib.full_reverse.FullReverse,
           "reader.early-window": contrib.early_window.EarlyWindow}


# ----------------------------------------------------------------------------
# Configurations
# ----------------------------------------------------------------------------
def benchmark_configuration(kind, world):
    """C6 of Chapter 16, or C6 with one module replaced by a contribution."""
    task = bm.operation(world, "forecast")
    demand = ("cart.task-plan", {"task": task})
    control = ("bench.constant-command",)
    if kind == "full-reverse":
        control = ("reader.full-reverse",)
    elif kind == "early-window":
        demand = ("reader.early-window", {"task": task, "early": EARLY})
    modules = {"control": control, "proposal": ("cart.naive",), "robot": ("cart.robot",),
               "world": ("cart.track",), "capability": cb.capability("survey", world),
               "demand": demand, "task": task, "evaluator": bm.Metrics(world, task)}
    return allovi.build(f"{kind}-{world.seed}", ce.T, modules)


def chapter3_configuration(kind, announce=140, onset=200):
    """The forecasting run of Chapter 3 (the reference configuration of the
    check suite of Section 10.3), with the operation announced in period
    'announce' for period 'onset', and with one module replaced."""
    task = cart.modules.Operation(announce=announce, onset=onset)
    parts = {"control": ("cart.constant-command",), "proposal": ("cart.naive",),
             "capability": ("cart.forecasting",), "demand": ("cart.task-plan", {"task": task}),
             "task": task, "robot": ("cart.robot",), "world": ("cart.track",),
             "evaluator": ("cart.metrics", {"task": task})}
    if kind == "full-reverse":
        parts["control"] = ("reader.full-reverse",)
    elif kind == "early-window":
        parts["demand"] = ("reader.early-window", {"task": task, "early": EARLY})
    return allovi.build(f"chapter3-{kind}", ce.T, parts), task


def run(spec):
    """One configuration in one world of the family."""
    kind, seed = spec
    world = bw.World(seed)
    cfg = benchmark_configuration(kind, world)
    ce.RAMP = world.ramp
    try:
        rec = allovi.run(cfg, PERIODS, strict=True)
        m = cfg.evaluator.evaluate(rec)
    finally:
        ce.RAMP = cb.RAMP_CH3
    m.update(kind=kind, world=seed)
    return m


# ----------------------------------------------------------------------------
# The steps of the path
# ----------------------------------------------------------------------------
def registration_facts():
    out = {}
    for label, role, name in CONTRIBUTIONS:
        cls = CLASSES[name]
        known = name in allovi.REGISTRY.names(role)
        out[name] = (f"role {cls.ROLE}; contract {cls.CONTRACT}; "
                     f"{'registered' if known else 'not registered'}")
    return out


def check_facts():
    """The check suite of Section 10.3, each module in its role in the
    configuration of the forecasting run of Chapter 3."""
    out = {}
    for label, role, name in CONTRIBUTIONS:
        results = checks.check(lambda: chapter3_configuration(label)[0], 220, cart.fmt)
        out[name] = checks.summary(results)
        failed = [r.line() for r in results if r.status != checks.PASSED]
        if failed:
            out[f"{name}: checks not passed"] = "; ".join(failed)
    return out


def chapter3_facts():
    """The world of Chapter 3: the shortest lead time of the full reverse, and
    the early window with the schedule of Chapter 3."""
    out = {}
    res = []
    for lead in (21, 20, 19, 18):
        for kind in ("C6", "full-reverse"):
            cfg, task = chapter3_configuration(kind, announce=200 - lead, onset=200)
            allovi.run(cfg, 220, strict=True)
            res.append((kind, lead, task.begun == 200))
    for kind in ("C6", "full-reverse"):
        out[f"{kind}: lead times in periods at which the operation begins on time; not"] = (
            ", ".join(str(lead) for k, lead, ok in res if k == kind and ok) + "; "
            + (", ".join(str(lead) for k, lead, ok in res if k == kind and not ok) or "none"))
    for kind in ("C6", "early-window"):
        cfg, task = chapter3_configuration(kind)
        rec = allovi.run(cfg, 220, strict=True)
        states = [x.state for x in rec.periods] + [rec.final_state]
        back = next(x.k for x in rec.periods if x.k >= 140 and x.state[1] < 0) - 1
        out[f"{kind}, the schedule of Chapter 3: begun; the back-away from period; periods in "
            f"the work zone after the announcement"] = (
            f"{task.begun}; {back}; "
            f"{sum(1 for s in states[140:200] if s[0] >= ce.WORK_ZONE)}")
    return out


def proportion(x, n):
    lo, hi = stats.clopper_pearson(x, n)
    return f"{x} of {n} ({stats.percent(x, n)} %; {stats.interval_percent(lo, hi)})"


def median_text(values, unit=" s", nd=1):
    if not values:
        return "none"
    med, lo, hi, cover = stats.median_interval(values)
    span = f"{float(lo):.{nd}f} to {float(hi):.{nd}f}" if cover is not None else "no interval"
    return (f"{float(med):.{nd}f}{unit} (95 % interval {span}; range "
            f"{float(min(values)):.{nd}f} to {float(max(values)):.{nd}f}; n = {len(values)})")


def configuration_facts(kind, runs, seeds, valid):
    vs = [runs[(kind, s)] for s in valid]
    ms = [runs[(kind, s)] for s in seeds]
    secs = lambda n: Fr(n) * ce.T
    return {
        "valid worlds: the operation on time": proportion(
            sum(1 for m in vs if m["delay"] == 0), len(vs)),
        "valid worlds: the operation late, by [s]": median_text(
            [secs(m["delay"]) for m in vs if m["delay"] not in (None, 0)]),
        "valid worlds: the operation never begun": f"{sum(1 for m in vs if m['begun'] is None)} "
                                                  f"of {len(vs)}",
        "valid worlds: arrival in the work zone [s]": median_text(
            [secs(m["arrival"]) for m in vs if m["arrival"] is not None]),
        "valid worlds: time in the work zone [s]": median_text(
            [secs(m["in_work_zone"]) for m in vs]),
        "all worlds: runs with an invalid executed label; with an alarm; restarts; violations": (
            f"{sum(1 for m in ms if m['invalid'])}; {sum(1 for m in ms if m['alarms'])}; "
            f"{sum(len(m['restarts']) for m in ms)}; {sum(1 for m in ms if m['violation'])} "
            f"of {len(ms)}"),
        "all worlds: the least true clearance in runs whose executed labels are valid [m]":
            f"{float(min(m['least_clearance'] for m in ms if not m['invalid'])):.2f}",
    }


class Rule:
    """Cases and exceptions of one hypothesis, as HYPOTHESES.md states it."""

    def __init__(self):
        self.cases, self.exceptions = 0, []

    def case(self, ok, text):
        self.cases += 1
        if not ok:
            self.exceptions.append(text)

    def text(self):
        exc = self.exceptions
        return (f"{'yes' if not exc else 'no'}; {self.cases}; "
                f"{'; '.join(exc[:6]) or 'none'}"
                + (f"; and {len(exc) - 6} more" if len(exc) > 6 else ""))


def hypotheses(runs, worlds, valid, ch3):
    """The decision rules of F1 to F4 and E1 to E5 (HYPOTHESES.md)."""
    out = {}
    for hid, kind in (("F1", "full-reverse"), ("E1", "early-window")):
        r = Rule()
        for s in sorted(worlds):
            m = runs[(kind, s)]
            if m["invalid"]:
                continue
            r.case(not m["alarms"] and m["least_clearance"] >= ce.EPS_MIN,
                   f"world {s}: alarms {m['alarms']}, least clearance "
                   f"{float(m['least_clearance']):.3f} m")
        out[f"{hid} (the certificate): supported; cases; exceptions"] = r.text()
    r = Rule()
    ok, notok = ch3["full-reverse"]
    r.case(19 in ok and 18 in notok,
           f"the world of Chapter 3: on time with {ok}, not with {notok}")
    for s in valid:
        w = worlds[s]
        if w.slip == 0 and w.lead in (19, 20):
            a, b = runs[("full-reverse", s)], runs[("C6", s)]
            r.case(a["delay"] == 0 and b["delay"] != 0,
                   f"world {s} (lead {w.lead}): full reverse delay {a['delay']}, C6 {b['delay']}")
    out["F2 (the shorter lead time): supported; cases; exceptions"] = r.text()
    r = Rule()
    for s in valid:
        a, b = runs[("full-reverse", s)], runs[("C6", s)]
        no_later = b["begun"] is None or (a["begun"] is not None and a["begun"] <= b["begun"])
        r.case(no_later and a["arrival"] == b["arrival"],
               f"world {s}: begun {a['begun']} against {b['begun']}, arrival {a['arrival']} "
               f"against {b['arrival']}")
    out["F3 (no loss): supported; cases; exceptions"] = r.text()
    r = Rule()
    for s in valid:
        w = worlds[s]
        if w.slip == 0 and w.lead >= 21:
            a, b = runs[("full-reverse", s)], runs[("C6", s)]
            d = a["in_work_zone"] - b["in_work_zone"]
            r.case(a["delay"] == 0 and b["delay"] == 0 and d in (1, 2),
                   f"world {s}: delays {a['delay']}, {b['delay']}; periods in the work zone "
                   f"more: {d}")
    out["F4 (later preparation): supported; cases; exceptions"] = r.text()
    r = Rule()
    for s in valid:
        w = worlds[s]
        if -EARLY <= w.slip <= -1 and w.lead >= 31:
            a, b = runs[("early-window", s)], runs[("C6", s)]
            r.case(a["delay"] == 0 and b["delay"] != 0,
                   f"world {s} (slip {w.slip}): early window delay {a['delay']}, C6 {b['delay']}")
    out["E2 (an earlier onset): supported; cases; exceptions"] = r.text()
    r = Rule()
    for s in valid:
        w = worlds[s]
        if w.slip == 0 and w.lead >= 31:
            a, b = runs[("early-window", s)], runs[("C6", s)]
            d = b["in_work_zone"] - a["in_work_zone"]
            r.case(a["delay"] == 0 and b["delay"] == 0 and d == EARLY,
                   f"world {s}: delays {a['delay']}, {b['delay']}; periods in the work zone "
                   f"fewer: {d}")
    out["E3 (the price): supported; cases; exceptions"] = r.text()
    r = Rule()
    for s in valid:
        w = worlds[s]
        if w.slip == 0 and 21 <= w.lead <= 30:
            a, b = runs[("early-window", s)], runs[("C6", s)]
            later = a["begun"] is None or (b["begun"] is not None and a["begun"] > b["begun"])
            r.case(later and b["delay"] == 0,
                   f"world {s} (lead {w.lead}): begun {a['begun']} against {b['begun']}")
    out["E4 (a window that cannot be met): supported; cases; exceptions"] = r.text()
    r = Rule()
    for s in sorted(worlds):
        a, b = runs[("early-window", s)], runs[("C6", s)]
        r.case(a["arrival"] == b["arrival"],
               f"world {s}: arrival {a['arrival']} against {b['arrival']}")
    out["E5 (the approach unchanged): supported; cases; exceptions"] = r.text()
    return out


def paired_facts(runs, valid):
    """Paired differences over the valid worlds, contribution less C6."""
    out = {}
    for kind in ("full-reverse", "early-window"):
        earlier = later = same = 0
        dz = {}
        for s in valid:
            a, b = runs[(kind, s)], runs[("C6", s)]
            ab, bb = a["begun"], b["begun"]
            if ab == bb:
                same += 1
            elif bb is None or (ab is not None and ab < bb):
                earlier += 1
            else:
                later += 1
            x = a["in_work_zone"] - b["in_work_zone"]
            dz[x] = dz.get(x, 0) + 1
        out[f"{kind}: the operation begun earlier; later; in the same period as C6 (valid worlds)"] = (
            f"{earlier}; {later}; {same}")
        out[f"{kind}: periods in the work zone less C6's: value (valid worlds)"] = ", ".join(
            f"{x} ({n})" for x, n in sorted(dz.items()))
    return out


def world_facts(worlds, valid):
    ws = [worlds[s] for s in sorted(worlds)]
    return {
        "worlds; valid; unreliable surveys": (f"{len(ws)}; {len(valid)}; "
                                              f"{sum(1 for w in ws if w.unreliable)}"),
        "valid worlds: lead times of 31 periods or more; 21 to 30; 19 or 20; 18 or fewer": "; ".join(
            str(sum(1 for s in valid if lo <= worlds[s].lead <= hi))
            for lo, hi in ((31, 99), (21, 30), (19, 20), (0, 18))),
        "valid worlds: onsets that slip later; earlier; not": (
            f"{sum(1 for s in valid if worlds[s].slip > 0)}; "
            f"{sum(1 for s in valid if worlds[s].slip < 0)}; "
            f"{sum(1 for s in valid if worlds[s].slip == 0)}"),
    }


def write_rows(path, runs, worlds):
    cols = ["kind", "world", "valid", "lead", "slip", "announce", "onset_forecast", "onset",
            "arrival", "in_work_zone", "begun", "delay", "overloads", "invalid", "alarms",
            "restarts", "least_clearance"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for (kind, s), m in sorted(runs.items(), key=lambda x: (KINDS.index(x[0][0]), x[0][1])):
            w = worlds[s]
            wr.writerow(dict(
                kind=kind, world=s, valid="yes" if cb.valid(w) else "no", lead=w.lead,
                slip=w.slip, announce=w.announce, onset_forecast=w.onset_forecast,
                onset=w.onset, arrival="" if m["arrival"] is None else m["arrival"],
                in_work_zone=m["in_work_zone"], begun="" if m["begun"] is None else m["begun"],
                delay="" if m["delay"] is None else m["delay"],
                overloads=" ".join(map(str, m["overloads"])), invalid=m["invalid"],
                alarms=" ".join(map(str, m["alarms"])),
                restarts=" ".join(map(str, m["restarts"])),
                least_clearance=f"{float(m['least_clearance']):.4f}"))


def main():
    ap = argparse.ArgumentParser(description="Chapter 17: the path of a contribution")
    ap.add_argument("--check", action="store_true", help="check every value of the report")
    ap.add_argument("--jobs", type=int, default=2, help="processes (default 2)")
    args = ap.parse_args()
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    facts = {"registration": registration_facts(), "the check suite": check_facts()}
    c3 = chapter3_facts()
    facts["the world of Chapter 3"] = c3
    ch3 = {}
    for kind in ("C6", "full-reverse"):
        ok, notok = c3[f"{kind}: lead times in periods at which the operation begins on time; "
                       f"not"].split("; ")
        ch3[kind] = ([int(x) for x in ok.split(", ") if x],
                     [int(x) for x in notok.split(", ") if x and x != "none"])
    worlds = {s: bw.World(s) for s in FAMILY}
    valid = sorted(s for s, w in worlds.items() if cb.valid(w))
    specs = [(k, s) for k in KINDS for s in sorted(worlds)]
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            out = pool.map(run, specs, chunksize=4)
    else:
        out = [run(x) for x in specs]
    runs = {(m["kind"], m["world"]): m for m in out}
    facts["the family"] = world_facts(worlds, valid)
    for kind in KINDS:
        facts[f"{kind}"] = configuration_facts(kind, runs, sorted(worlds), valid)
    facts["the hypotheses"] = hypotheses(runs, worlds, valid, ch3)
    facts["paired with C6"] = paired_facts(runs, valid)
    write_rows(os.path.join(out_dir, "contribution-runs.csv"), runs, worlds)
    lines, failed = [], []
    for group, items in facts.items():
        lines.append(f"[{group}]")
        for key, val in items.items():
            lines.append(f"  {key} = {val}")
            want = EXPECTED.get(group, {}).get(key)
            if args.check and want is not None and want != str(val):
                failed.append(f"{group}.{key}: expected {want!r}, got {val!r}")
    if args.check:
        failed += [f"{group}.{key}: not computed" for group, items in EXPECTED.items()
                   for key in items if key not in facts.get(group, {})]
        n = sum(len(v) for v in EXPECTED.values())
        lines.append(f"[check] {n} values: " +
                     ("all as expected" if not failed and n else "FAILED: " + "; ".join(failed)))
    with open(os.path.join(out_dir, "contribution-summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if failed:
        raise SystemExit(1)


EXPECTED = {   # every value of the summary but the versions of the libraries
 'registration': {'reader.full-reverse': 'role control; contract 1.0; registered',
                  'reader.early-window': 'role demand; contract 1.0; registered'},
 'the check suite': {'reader.full-reverse': '13 passed, 0 failed, 0 not applicable',
                     'reader.early-window': '13 passed, 0 failed, 0 not applicable'},
 'the world of Chapter 3': {'C6: lead times in periods at which the operation begins on time; not': '21; '
                                                                                                    '20, '
                                                                                                    '19, '
                                                                                                    '18',
                            'full-reverse: lead times in periods at which the operation begins on time; not': '21, '
                                                                                                              '20, '
                                                                                                              '19; '
                                                                                                              '18',
                            'C6, the schedule of Chapter 3: begun; the back-away from period; periods in the work zone after the announcement': '200; '
                                                                                                                                                '180; '
                                                                                                                                                '55',
                            'early-window, the schedule of Chapter 3: begun; the back-away from period; periods in the work zone after the announcement': '200; '
                                                                                                                                                          '170; '
                                                                                                                                                          '45'},
 'the family': {'worlds; valid; unreliable surveys': '100; 95; 19',
                'valid worlds: lead times of 31 periods or more; 21 to 30; 19 or 20; 18 or fewer': '66; '
                                                                                                   '14; '
                                                                                                   '1; '
                                                                                                   '14',
                'valid worlds: onsets that slip later; earlier; not': '12; 13; 70'},
 'C6': {'valid worlds: the operation on time': '69 of 95 (73 %; 62.5 to 81.3 %)',
        'valid worlds: the operation late, by [s]': '2.1 s (95 % interval 0.5 to 2.1; range 0.1 to '
                                                    '2.7; n = 26)',
        'valid worlds: the operation never begun': '0 of 95',
        'valid worlds: arrival in the work zone [s]': '8.9 s (95 % interval 8.8 to 8.9; range 8.4 '
                                                      'to 9.2; n = 95)',
        'valid worlds: time in the work zone [s]': '13.0 s (95 % interval 12.5 to 13.2; range 10.6 '
                                                   'to 16.1; n = 95)',
        'all worlds: runs with an invalid executed label; with an alarm; restarts; violations': '5; '
                                                                                                '5; '
                                                                                                '2; '
                                                                                                '0 '
                                                                                                'of '
                                                                                                '100',
        'all worlds: the least true clearance in runs whose executed labels are valid [m]': '0.50'},
 'full-reverse': {'valid worlds: the operation on time': '69 of 95 (73 %; 62.5 to 81.3 %)',
                  'valid worlds: the operation late, by [s]': '2.1 s (95 % interval 0.5 to 2.1; '
                                                              'range 0.1 to 2.2; n = 26)',
                  'valid worlds: the operation never begun': '0 of 95',
                  'valid worlds: arrival in the work zone [s]': '8.9 s (95 % interval 8.8 to 8.9; '
                                                                'range 8.4 to 9.2; n = 95)',
                  'valid worlds: time in the work zone [s]': '13.0 s (95 % interval 12.5 to 13.2; '
                                                             'range 10.6 to 16.1; n = 95)',
                  'all worlds: runs with an invalid executed label; with an alarm; restarts; violations': '5; '
                                                                                                          '5; '
                                                                                                          '2; '
                                                                                                          '0 '
                                                                                                          'of '
                                                                                                          '100',
                  'all worlds: the least true clearance in runs whose executed labels are valid [m]': '0.50'},
 'early-window': {'valid worlds: the operation on time': '71 of 95 (75 %; 64.7 to 83.1 %)',
                  'valid worlds: the operation late, by [s]': '6.3 s (95 % interval 4.2 to 6.3; '
                                                              'range 2.1 to 6.4; n = 24)',
                  'valid worlds: the operation never begun': '0 of 95',
                  'valid worlds: arrival in the work zone [s]': '8.9 s (95 % interval 8.8 to 8.9; '
                                                                'range 8.4 to 9.2; n = 95)',
                  'valid worlds: time in the work zone [s]': '12.2 s (95 % interval 11.7 to 12.9; '
                                                             'range 9.6 to 18.3; n = 95)',
                  'all worlds: runs with an invalid executed label; with an alarm; restarts; violations': '5; '
                                                                                                          '5; '
                                                                                                          '2; '
                                                                                                          '0 '
                                                                                                          'of '
                                                                                                          '100',
                  'all worlds: the least true clearance in runs whose executed labels are valid [m]': '0.50'},
 'the hypotheses': {'F1 (the certificate): supported; cases; exceptions': 'yes; 95; none',
                    'E1 (the certificate): supported; cases; exceptions': 'yes; 95; none',
                    'F2 (the shorter lead time): supported; cases; exceptions': 'yes; 1; none',
                    'F3 (no loss): supported; cases; exceptions': 'yes; 95; none',
                    'F4 (later preparation): supported; cases; exceptions': 'no; 58; world 16101: '
                                                                            'delays 0, 0; periods '
                                                                            'in the work zone '
                                                                            'more: 0; world 16102: '
                                                                            'delays 0, 0; periods '
                                                                            'in the work zone '
                                                                            'more: 0; world 16105: '
                                                                            'delays 0, 0; periods '
                                                                            'in the work zone '
                                                                            'more: 0; world 16106: '
                                                                            'delays 0, 0; periods '
                                                                            'in the work zone '
                                                                            'more: 0; world 16110: '
                                                                            'delays 0, 0; periods '
                                                                            'in the work zone '
                                                                            'more: 0; world 16111: '
                                                                            'delays 0, 0; periods '
                                                                            'in the work zone '
                                                                            'more: 0; and 52 more',
                    'E2 (an earlier onset): supported; cases; exceptions': 'yes; 11; none',
                    'E3 (the price): supported; cases; exceptions': 'yes; 47; none',
                    'E4 (a window that cannot be met): supported; cases; exceptions': 'no; 11; '
                                                                                      'world 16101 '
                                                                                      '(lead 27): '
                                                                                      'begun 218 '
                                                                                      'against '
                                                                                      '218; world '
                                                                                      '16102 (lead '
                                                                                      '28): begun '
                                                                                      '235 against '
                                                                                      '235; world '
                                                                                      '16128 (lead '
                                                                                      '27): begun '
                                                                                      '213 against '
                                                                                      '213; world '
                                                                                      '16140 (lead '
                                                                                      '30): begun '
                                                                                      '205 against '
                                                                                      '205',
                    'E5 (the approach unchanged): supported; cases; exceptions': 'yes; 100; none'},
 'paired with C6': {'full-reverse: the operation begun earlier; later; in the same period as C6 (valid worlds)': '1; '
                                                                                                                 '0; '
                                                                                                                 '94',
                    "full-reverse: periods in the work zone less C6's: value (valid worlds)": '0 '
                                                                                              '(95)',
                    'early-window: the operation begun earlier; later; in the same period as C6 (valid worlds)': '11; '
                                                                                                                 '24; '
                                                                                                                 '60',
                    "early-window: periods in the work zone less C6's: value (valid worlds)": '-10 '
                                                                                              '(66), '
                                                                                              '-9 '
                                                                                              '(1), '
                                                                                              '-7 '
                                                                                              '(2), '
                                                                                              '-6 '
                                                                                              '(3), '
                                                                                              '-5 '
                                                                                              '(1), '
                                                                                              '-2 '
                                                                                              '(1), '
                                                                                              '10 '
                                                                                              '(1), '
                                                                                              '14 '
                                                                                              '(3), '
                                                                                              '15 '
                                                                                              '(2), '
                                                                                              '16 '
                                                                                              '(2), '
                                                                                              '17 '
                                                                                              '(2), '
                                                                                              '18 '
                                                                                              '(1), '
                                                                                              '20 '
                                                                                              '(2), '
                                                                                              '29 '
                                                                                              '(5), '
                                                                                              '30 '
                                                                                              '(3)'}}


if __name__ == "__main__":
    main()
