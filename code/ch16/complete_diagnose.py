#!/usr/bin/env python3
"""
The diagnosis of the complete case's failed hypotheses and paired
differences: exploratory, written after the evaluation.

Physical AI, Together, Chapter 16 (Section 16.4). Not part of the protocol
of complete/PROTOCOL.md: it was written after the evaluation's results were
known, it tests nothing, and its findings are explanations of the recorded
runs, not results under the protocol.

It reruns offline, exactly as complete_evaluate.py did (the runs are
deterministic, and every rerun is checked against its row of
results/complete-runs.csv), two groups of runs:
  1. the worlds in which H3 or H4 failed: A-learner and A-rule with the
     package of the failure;
  2. the valid worlds in which F-preview began the operation on time and
     A-rule did not: both configurations, with the package of the
     difference.
For each run it reports the true state at the period of the announcement,
k0 = onset forecast - lead (the first period in which the demand predictor
issues the requirement), the largest speed in the ten periods before it,
the first period from k0 on in which the cart moves away from the wall
(true speed below -1 cm/s), the true and the forecast onset, and when the
operation began; for the configurations with the demand predictor, the
level that the committed plan holds from the pair of the forecast onset
on, at the announcement and at best before the forecast onset, against
the required level, the first period whose committed plan holds the
requirement, and the number of periods from the announcement to the
forecast onset in which the governor committed the fallback; for every
run, the proposals and the committed ambitions in the three periods before
the announcement, the level in force at the forecast and at the true onset,
and the fastest backward speed before the true onset.

It then reads results/complete-realtime-periods.csv (no rerun) and lists
  3. every period of the real-time phase that executed the fallback because
     its plan arrived late, and every period whose evaluation the deadline
     cut, with the delays of the period in which the plan was decided: when
     the plant published the measurement and when the governor received it,
     after the boundary, the plant's computation of the period, the
     decision, and the slack left before the next boundary;
and counts, over all periods, how often each delay exceeded 10, 50 and
94 ms (the deadline lies 6 ms before the boundary).

Last, it reruns offline the 60 runs of the stress test S1 (sliding friction
0.20, A-rule, both packages), as complete_stress.py ran them, and reports
  4. where the cart went after its first alarm, its fastest true speeds,
     whether it went back behind the start of the crest, how many periods
     ended without a contact of its wheels with the ground, and its state
     at the end of the run.
It writes results/complete-diagnosis.txt.

Usage
    python3 complete_diagnose.py

Requires MuJoCo 3.16.0 and NumPy. A few minutes with two processes.
"""
import csv
import math
import os
import statistics
import sys
from multiprocessing import Pool

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import complete                                       # noqa: E402,F401
from complete import config, model as md, worlds      # noqa: E402
from complete_evaluate import RESULTS, check_frozen, frozen_offset   # noqa: E402

BACK = -0.01          # [m/s]: moving away from the wall
ICY = 9.81 * (0.20 * math.cos(math.radians(9.5)) - math.sin(math.radians(9.5)))
                      # [m/s^2]: the braking that the icy slope allows (S1)
CREST = 1.66          # [m]: the crest's length along the ground (Section 11.1); the slope
                      # begins this far beyond the start of the crest


def job(spec):
    seed, name, pkg, bounds, offset, tables = spec
    config.LEARNERS.update(tables)
    world = worlds.World(seed)
    rec, m, _ = config.run(world, name, pkg, bounds, offset)
    states = [x.state for x in rec.periods] + [rec.final_state]
    k0 = world.announce
    p0, v0 = float(states[k0][0]), float(states[k0][1])
    vmax = max(abs(float(s[1])) for s in states[max(0, k0 - 10):k0 + 1])
    back = next((k for k in range(k0, len(states)) if float(states[k][1]) < BACK), None)
    per, fo = rec.periods, world.onset_forecast

    def at_onset(k):
        """The level that the plan committed in period k holds from the pair of
        the forecast onset on, and the level that the requirement asks for, if
        the demand predictor issued one (A-configurations)."""
        pr = per[k]
        d, req = pr.decision, pr.requirement
        if d is None or d.profile is None or req is None or req.level <= md.EPS_MIN:
            return None
        return float(d.profile.at(req.onset)), float(req.level)

    window = [at_onset(k) for k in range(k0, min(fo, len(per)))]
    held = [x for x in window if x is not None]
    first_held = next((k0 + i for i, x in enumerate(window) if x is not None and x[0] >= x[1]),
                      None)
    fallbacks = sum(1 for k in range(k0, min(fo, len(per)))
                    if per[k].decision is not None and per[k].decision.selected == 0)
    pre = [(per[k].proposal, str(per[k].decision.ambition) if per[k].decision else None)
           for k in range(max(0, k0 - 3), k0)]
    vmin = min(float(s[1]) for s in states[k0:min(world.onset, len(states) - 1) + 1])

    def lv(k):
        return None if k >= len(per) else round(float(per[k].level), 3)
    return dict(world=seed, config=name, package=pkg, lead=world.lead, slip=world.slip,
                announce=k0, onset_forecast=fo, onset=world.onset,
                begun=m["begun"], on_time=m["on_time"], arrival=m["arrival"],
                p_at_announce=round(p0, 4), v_at_announce=round(v0, 4),
                vmax_before=round(vmax, 4), first_back=back,
                back_after_announce=None if back is None else back - k0,
                required=None if not held else held[0][1],
                best_at_onset=None if not held else round(max(x[0] for x in held), 3),
                at_onset_announce=None if window[0] is None else round(window[0][0], 3),
                first_held=first_held, fallbacks=fallbacks, pre=pre,
                level_forecast=lv(fo), level_onset=lv(world.onset), vmin=round(vmin, 3))


def job_s1(spec):
    """One run of the stress test S1 (sliding friction 0.20), as complete_stress.py
    runs it, with what its summary does not report: where the cart went after
    its first alarm, how fast, and whether its wheels kept the ground."""
    seed, pkg, bounds, offset = spec
    world = worlds.World(seed)
    rec, m, sim = config.run(world, "A-rule", pkg, bounds, offset, record=True,
                             design_kw=dict(friction=0.20))
    states = [x.state for x in rec.periods] + [rec.final_state]
    first = rec.alarms[0] if rec.alarms else None
    after = states[first:] if first is not None else []
    per = sim.periods
    # the trace: (t, tau, p, v, l, p_meas, v_meas, l_meas, u, ell, contacts, front, rear)
    tr = sim.plant.trace
    slope = world.ramp_start + CREST
    end = (first if first is not None else len(rec.periods)) * 100
    u_slope = [r[8] for r in tr[:end] if r[2] >= slope]
    loads = [r[11] + r[12] for r in tr]
    return dict(world=seed, package=pkg, alarms=len(rec.alarms), first_alarm=first,
                u_min_slope=None if not u_slope else round(min(u_slope), 3),
                load_median=round(statistics.median(loads[:5000]), 1),
                load_max=round(max(loads), 1),
                p_at_alarm=None if first is None else round(float(states[first][0]), 3),
                v_at_alarm=None if first is None else round(float(states[first][1]), 3),
                p_min_after=None if not after else round(min(float(s[0]) for s in after), 3),
                behind_crest=bool(after) and min(float(s[0]) for s in after) < world.ramp_start,
                v_min=round(min(float(s[1]) for s in states), 3),
                v_max=round(max(float(s[1]) for s in states), 3),
                airborne=sum(1 for s in per if s["contacts"] == 0),
                p_end=round(float(states[-1][0]), 3), v_end=round(float(states[-1][1]), 3),
                rest_periods=m["rest_feedback"], max_w=round(m["max_w"], 3),
                clearance=round(m["least_clearance"], 4), violation=bool(m["violation"]))


def s1_lines(rows):
    """[4]: S1 by package, from job_s1."""
    out = ["[4] stress test S1 (sliding friction 0.20), A-rule, rerun offline: after the "
           "first alarm, the rest feedback (no guarantee) acts until a restart"]
    for pkg in config.PACKAGES:
        rs = [r for r in rows if r["package"] == pkg]
        al = [r for r in rs if r["alarms"]]
        out.append(f"  {pkg}: runs {len(rs)}; runs with an alarm {len(al)}; violations "
                   f"{sum(r['violation'] for r in rs)}; least clearance "
                   f"{min(r['clearance'] for r in rs):.4f} m")
        us = [r["u_min_slope"] for r in rs if r["u_min_slope"] is not None]
        out.append(f"    strongest commanded braking on the slope before the first alarm (or in "
                   f"the whole run, without one) {min(us):+.3f} m/s^2 (the icy slope allows "
                   f"g(0.20 cos 9.5 deg - sin 9.5 deg) = {ICY:.3f}; per run from {max(us):+.3f}); runs "
                   f"whose command on the slope went below -{ICY:.3f} m/s^2: "
                   f"{sum(1 for u in us if u < -ICY)} of {len(us)}")
        out.append(f"    total load of the wheels: median over the first 50 periods "
                   f"{min(r['load_median'] for r in rs):.0f} to {max(r['load_median'] for r in rs):.0f} N; "
                   f"largest in the run {min(r['load_max'] for r in rs):.0f} to "
                   f"{max(r['load_max'] for r in rs):.0f} N")
        if not al:
            out.append(f"    fastest backward and forward true speeds {min(r['v_min'] for r in rs):+.3f}, "
                       f"{max(r['v_max'] for r in rs):+.3f} m/s; periods ending without a "
                       f"contact of the wheels {sum(r['airborne'] for r in rs)}")
            continue
        out.append(f"    first alarm in periods {min(r['first_alarm'] for r in al)} to "
                   f"{max(r['first_alarm'] for r in al)}, at {min(r['p_at_alarm'] for r in al):.2f} "
                   f"to {max(r['p_at_alarm'] for r in al):.2f} m, true speed "
                   f"{min(r['v_at_alarm'] for r in al):+.3f} to {max(r['v_at_alarm'] for r in al):+.3f} m/s")
        out.append(f"    runs in which the cart went back behind the start of the crest after "
                   f"the alarm {sum(r['behind_crest'] for r in al)}; least position after the "
                   f"alarm {min(r['p_min_after'] for r in al):.2f} m; fastest backward true speed "
                   f"{min(r['v_min'] for r in al):+.2f} m/s (runs: "
                   f"{sorted(r['v_min'] for r in al)[:3]} ... {sorted(r['v_min'] for r in al)[-3:]})")
        out.append(f"    runs with periods ending without a contact of the wheels "
                   f"{sum(1 for r in al if r['airborne'])}, {sum(r['airborne'] for r in al)} "
                   f"periods in all; at the end of the run: position "
                   f"{min(r['p_end'] for r in al):.2f} to {max(r['p_end'] for r in al):.2f} m, speed "
                   f"{min(r['v_end'] for r in al):+.3f} to {max(r['v_end'] for r in al):+.3f} m/s; "
                   f"periods of the rest feedback {min(r['rest_periods'] for r in al)} to "
                   f"{max(r['rest_periods'] for r in al)}")
    return out


def main():
    check_frozen()
    bounds, q_hat = md.bounds(True), frozen_offset()
    config.load_learners(RESULTS)
    tables = dict(config.LEARNERS)
    with open(os.path.join(RESULTS, "complete-runs.csv")) as fh:
        rows = list(csv.DictReader(fh))
    by = {(r["config"], r["package"], int(r["world"])): r for r in rows}
    valid = sorted({int(r["world"]) for r in rows if float(r["score"]) <= q_hat + 1e-12})
    from complete_evaluate import back_periods
    n_back = {p: back_periods(p, bounds) for p in config.PACKAGES}
    group1, group2 = [], []
    for p in config.PACKAGES:
        for s in valid:
            rule, learner = by[("A-rule", p, s)], by[("A-learner", p, s)]
            h4 = rule["on_time"] == "True" and learner["on_time"] != "True"
            h3 = (learner["on_time"] != "True" and int(learner["slip"]) >= 0
                  and int(learner["lead"]) >= n_back[p] + 2)
            if h3 or h4:
                group1.append((s, p))
            if by[("F-preview", p, s)]["on_time"] == "True" and rule["on_time"] != "True":
                group2.append((s, p))
    specs = ([(s, c, p, bounds, q_hat, tables) for s, p in group1 for c in ("A-rule", "A-learner")]
             + [(s, c, p, bounds, q_hat, tables) for s, p in group2 for c in ("A-rule", "F-preview")])
    s1_specs = [(s, p, bounds, q_hat) for p in config.PACKAGES for s in worlds.PHASES["stress"]]
    with Pool(2) as pool:
        out = pool.map(job, specs, chunksize=1)
        s1 = pool.map(job_s1, s1_specs, chunksize=1)
    lines = ["Diagnosis of the complete case (complete_diagnose.py): exploratory, written "
             "after the evaluation; not a result under PROTOCOL.md",
             f"  N_back = {n_back}"]
    mismatch = [o for o in out if str(o["begun"]) != (by[(o["config"], o["package"], o["world"])]["begun"] or "None")]
    lines.append(f"  reruns that differ from complete-runs.csv in the period the operation began: "
                 f"{len(mismatch)}")
    for title, group, configs in (("[1] worlds of H3 and H4 (A-rule, A-learner)", group1,
                                   ("A-rule", "A-learner")),
                                  ("[2] F-preview on time, A-rule not", group2,
                                   ("A-rule", "F-preview"))):
        lines.append(title)
        for s, p in group:
            for c in configs:
                o = next(x for x in out if x["world"] == s and x["package"] == p and x["config"] == c)
                lines.append(
                    f"  {s} {p:4s} {c:9s} lead {o['lead']:2d} slip {o['slip']:+3d} announce "
                    f"{o['announce']} onset {o['onset']} (forecast {o['onset_forecast']}) begun "
                    f"{o['begun']}; at the announcement p {o['p_at_announce']:.3f} m, v "
                    f"{o['v_at_announce']:+.3f} m/s, |v| up to {o['vmax_before']:.3f} m/s in the "
                    f"ten periods before; backing away from period {o['first_back']} "
                    f"({o['back_after_announce']} after the announcement)")
                if o["required"] is not None:
                    lines.append(
                        f"      required level {o['required']:.2f} from the forecast onset; the "
                        f"committed plan held from it {o['at_onset_announce']} at the "
                        f"announcement and at best {o['best_at_onset']} before the forecast "
                        f"onset; first period holding the requirement {o['first_held']}")
                lines.append(
                    f"      proposals (committed ambition) in the three periods before the "
                    f"announcement {o['pre']}; fallbacks committed in {o['fallbacks']} of the "
                    f"{o['onset_forecast'] - o['announce']} periods from the announcement to the "
                    f"forecast onset; level in force at the forecast onset "
                    f"{o['level_forecast']}, at the true onset {o['level_onset']}; fastest "
                    f"backward speed before the onset {o['vmin']:+.3f} m/s")
    lines.extend(timing_lines())
    lines.extend(s1_lines(s1))
    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(RESULTS, "complete-diagnosis.txt"), "w") as fh:
        fh.write(text)


def timing_lines():
    """[3]: the real-time periods whose plan arrived late or whose evaluation
    the deadline cut, from results/complete-realtime-periods.csv."""
    path = os.path.join(RESULTS, "complete-realtime-periods.csv")
    if not os.path.exists(path):
        return ["[3] real time: results/complete-realtime-periods.csv not found"]
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    by = {(r["config"], r["package"], r["world"], int(r["k"])): r for r in rows}
    for r in rows:
        # the rest of the governor's period: receipt to sending, less the proposal and
        # the decision (labels, requirement, initial set, messages, and any time the
        # process was not running)
        r["other_ms"] = (1000 * md.T - float(r["slack_ms"]) - float(r["recv_ms"])
                         - float(r["proposal_ms"]) - float(r["decide_ms"]))
    keys = ("publish_late_ms", "recv_ms", "proposal_ms", "decide_ms", "other_ms", "slack_ms",
            "physics_ms")
    out = ["[3] real time: periods that executed the fallback because the plan arrived "
           "late, and periods whose evaluation the deadline cut",
           "    (for a fallback in period k, the times of period k-1, in which its plan was "
           "decided; for a cut, those of the period itself) [ms]: publication by the plant "
           "and receipt by the governor after the boundary; the proposal; the decision; the "
           "rest of the governor's period until the plan was sent; the slack left before the "
           "next boundary; the plant's computation of the period"]
    late = [r for r in rows if r["source"] == "fallback"]
    cut = [r for r in rows if r["cut"] == "True"]
    out.append(f"  periods = {len(rows)}; fallbacks after a late plan = {len(late)}; "
               f"evaluations cut = {len(cut)}")
    sent_late = [r for r in rows if float(r["slack_ms"]) < 0]
    out.append(f"  plans sent after the next boundary (negative slack) = {len(sent_late)}; for "
               f"each, the period that decided it and what the next period executed, with how "
               f"late the plant published that period's measurement")
    for r in sent_late:
        nxt = by.get((r["config"], r["package"], r["world"], int(r["k"]) + 1))
        vals = ", ".join(f"{k[:-3]} {float(r[k]):.1f}" for k in keys)
        tail = ("the run's end" if nxt is None else
                f"{nxt['source']} (plant {float(nxt['publish_late_ms']):.1f} ms late)")
        out.append(f"    {r['config']}-{r['package']} {r['world']} decided in {r['k']}: {vals}; "
                   f"evaluated {r['evaluated']}; period {int(r['k']) + 1} executed {tail}")
    for title, group, shift in (("  late plans", late, 1), ("  cut evaluations", cut, 0)):
        out.append(title)
        for r in group:
            prev = by.get((r["config"], r["package"], r["world"], int(r["k"]) - shift))
            if prev is None:
                out.append(f"    {r['config']}-{r['package']} {r['world']} period {r['k']}: "
                           f"no earlier period")
                continue
            vals = ", ".join(f"{k[:-3]} {float(prev[k]):.1f}" for k in keys)
            out.append(f"    {r['config']}-{r['package']} {r['world']} period {r['k']}"
                       f" (decided in {prev['k']}): {vals}; evaluated {prev['evaluated']}")
    out.append("  all periods: how many exceeded 10, 50 and 94 ms")
    for k in ("publish_late_ms", "recv_ms", "proposal_ms", "decide_ms", "other_ms",
              "physics_ms"):
        v = [float(r[k]) for r in rows]
        out.append(f"    {k[:-3]}: {sum(x > 10 for x in v)}, {sum(x > 50 for x in v)}, "
                   f"{sum(x > 94 for x in v)}; median {statistics.median(v):.3f}, "
                   f"largest {max(v):.3f}")
    return out


if __name__ == "__main__":
    main()
