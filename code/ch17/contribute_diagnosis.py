#!/usr/bin/env python3
"""
What the two worked contributions did where their hypotheses failed, or
where they did what no hypothesis stated (Section 17.1).

Physical AI, Together, Chapter 17.

Written after the first run of contribute.py, and therefore a diagnosis, not
a test: it reruns the worlds concerned and reports what the runs did. It
changes nothing in contribute.py, its hypotheses or its decision rules.

  * F2: the worlds of its subset in the family, and the one world in which
    the full reverse began the operation earlier than C6;
  * F4: in the worlds of its subset, when the governor stops following the
    proposal, when the cart first moves backwards, whether the full reverse
    is committed, and whether the executed commands differ from C6's;
  * E4: in the worlds of its subset, where the cart first comes to rest
    after the announcement, and whether the operation begins on time;
  * the early window's late runs: by lead time, and the postponements that
    each run needed;
  * the five worlds that are not valid: the alarms, the restarts, the
    operation and the least true clearance of each configuration.

Usage
    python3 contribute_diagnosis.py           # writes results/contribution-diagnosis.txt
                                              # and results/contribution-leads.csv
    python3 contribute_diagnosis.py --check   # also checks every value
    python3 contribute_diagnosis.py --jobs 1  # one process (default: up to two)

Python 3.8 or later; no other packages. About four minutes with two processes.
"""
import argparse
import os
import sys
from multiprocessing import Pool

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import contribute as co                                           # noqa: E402
import allovi                                                     # noqa: E402
import cart_benchmark as cb                                       # noqa: E402
from bench import worlds as bw                                    # noqa: E402
from bench.modules import TARGET                                  # noqa: E402  23.7 m
from cart.model import ce                                         # noqa: E402


def run(spec):
    """One configuration in one world: the metrics of contribute.py and what
    the diagnosis needs from the record."""
    kind, seed = spec
    world = bw.World(seed)
    cfg = co.benchmark_configuration(kind, world)
    ce.RAMP = world.ramp
    try:
        rec = allovi.run(cfg, co.PERIODS, strict=True)
        m = cfg.evaluator.evaluate(rec)
    finally:
        ce.RAMP = cb.RAMP_CH3
    a = world.announce
    states = [x.state for x in rec.periods] + [rec.final_state]
    normal = [x for x in rec.periods if x.governor == "normal"]
    first_modified = next((x.k for x in normal if x.k >= a and x.decision.ambition < 1), None)
    first_back = next((k for k in range(a, len(states)) if states[k][1] < 0), None)
    first_rest = None
    if first_back is not None:
        first_rest = next((k for k in range(first_back, len(states)) if states[k][1] == 0), None)
    until = m["begun"] if m["begun"] is not None else len(rec.periods)
    commands = [(x.k, x.plan.pairs[0][0].u if x.plan is not None and x.plan.pairs else None)
                for x in rec.periods if x.k < until]
    fr_committed = [x.k for x in normal
                    if len(x.decision.entries) == 8 and x.decision.selected == 7]
    m.update(kind=kind, world=seed, first_modified=first_modified, first_back=first_back,
             first_rest=first_rest,
             rest_place=None if first_rest is None else states[first_rest][0],
             commands=commands, fr_committed=fr_committed)
    return m


def values(counter):
    return ", ".join(f"{x} ({n})" for x, n in sorted(counter.items()))


def count(xs):
    out = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


def main():
    ap = argparse.ArgumentParser(description="Chapter 17: diagnosis of the worked contributions")
    ap.add_argument("--check", action="store_true", help="check every value")
    ap.add_argument("--jobs", type=int, default=2, help="processes (default 2)")
    args = ap.parse_args()
    target = TARGET
    worlds = {s: bw.World(s) for s in co.FAMILY}
    valid = sorted(s for s, w in worlds.items() if cb.valid(w))
    f4 = [s for s in valid if worlds[s].slip == 0 and worlds[s].lead >= 21]
    e4 = [s for s in valid if worlds[s].slip == 0 and 21 <= worlds[s].lead <= 30]
    specs = sorted({(k, s) for k in co.KINDS for s in worlds})
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            out = pool.map(run, specs, chunksize=4)
    else:
        out = [run(x) for x in specs]
    runs = {(m["kind"], m["world"]): m for m in out}
    facts = {}

    # F2 and the world in which the full reverse began earlier
    g = {}
    g["valid worlds with a lead time of 19 or 20 periods: world (slip)"] = ", ".join(
        f"{s} ({worlds[s].slip})" for s in valid if worlds[s].lead in (19, 20)) or "none"
    g["of these, without a slip (F2's subset in the family)"] = str(sum(
        1 for s in valid if worlds[s].lead in (19, 20) and worlds[s].slip == 0))
    for s in valid:
        a, b = runs[("full-reverse", s)], runs[("C6", s)]
        if a["begun"] != b["begun"]:
            w = worlds[s]
            g[f"world {s}, the full reverse begins earlier than C6: lead; slip; the forecast "
              f"onset; the true onset; begun, the full reverse and C6; overloads, C6"] = (
                f"{w.lead}; {w.slip}; {w.onset_forecast}; {w.onset}; {a['begun']}, {b['begun']}; "
                f"{' '.join(map(str, b['overloads'])) or 'none'}")
    facts["F2, the full reverse in the family"] = g

    # F4
    g = {"worlds of F4's subset": str(len(f4))}
    d_mod = [runs[("full-reverse", s)]["first_modified"] - runs[("C6", s)]["first_modified"]
             for s in f4]
    d_back = [runs[("full-reverse", s)]["first_back"] - runs[("C6", s)]["first_back"] for s in f4]
    g["the period in which the governor first commits an ambition below 1 after the "
      "announcement, the full reverse less C6: value (worlds)"] = values(count(d_mod))
    g["the period in which the cart first moves backwards after the announcement, the full "
      "reverse less C6: value (worlds)"] = values(count(d_back))
    committed = [len(runs[("full-reverse", s)]["fr_committed"]) for s in f4]
    g["worlds in which the governor commits the full reverse; periods in which it does, "
      "least and most"] = (f"{sum(1 for n in committed if n)}; "
                           f"{min(committed)} and {max(committed)}")
    first_fr = [runs[("full-reverse", s)]["fr_committed"][0] - runs[("C6", s)]["first_modified"]
                for s in f4 if runs[("full-reverse", s)]["fr_committed"]]
    g["its first commitment, less the period in which C6 first commits an ambition below 1: "
      "value (worlds)"] = values(count(first_fr))
    g["worlds in which an executed command differs from C6's before the operation begins"] = str(
        sum(1 for s in f4 if runs[("full-reverse", s)]["commands"] != runs[("C6", s)]["commands"]))
    facts["F4, the full reverse in the worlds without a slip and a lead time of 21 periods or more"] = g

    # E4
    g = {"worlds of E4's subset": str(len(e4))}
    for s in e4:
        e, c = runs[("early-window", s)], runs[("C6", s)]
        w = worlds[s]
        g[f"world {s}: lead; the early window's onset; the cart first at rest after the "
          f"announcement, period and place [m]; begun, the early window and C6"] = (
            f"{w.lead}; {w.onset_forecast - co.EARLY}; {e['first_rest']}, "
            f"{float(e['rest_place']):.4f}; {e['begun']}, {c['begun']}")
    before = [s for s in e4 if runs[("early-window", s)]["rest_place"] <= target]
    beyond = [s for s in e4 if runs[("early-window", s)]["rest_place"] > target]
    on_time = lambda ss: sum(1 for s in ss if runs[("early-window", s)]["delay"] == 0)
    g["first at rest at or before 23.7 m: worlds; their lead times; on time"] = (
        f"{len(before)}; {', '.join(str(worlds[s].lead) for s in before)}; {on_time(before)}")
    g["first at rest beyond 23.7 m: worlds; their lead times; on time"] = (
        f"{len(beyond)}; {', '.join(str(worlds[s].lead) for s in beyond)}; {on_time(beyond)}")
    facts["E4, the early window in the worlds without a slip and a lead time of 21 to 30 periods"] = g

    # The early window's late runs
    g = {}
    later = [s for s in valid if runs[("early-window", s)]["begun"] is not None
             and runs[("C6", s)]["begun"] is not None
             and runs[("early-window", s)]["begun"] > runs[("C6", s)]["begun"]]
    bands = (("20 or fewer", 0, 20), ("21 to 26", 21, 26), ("27 or more", 27, 99))
    g["valid worlds in which the early window begins the operation later than C6, by lead "
      "time: 20 or fewer; 21 to 26; 27 or more"] = "; ".join(
        str(sum(1 for s in later if lo <= worlds[s].lead <= hi)) for _, lo, hi in bands)
    g["of these, C6 on time"] = "; ".join(
        str(sum(1 for s in later if lo <= worlds[s].lead <= hi
                and runs[("C6", s)]["delay"] == 0)) for _, lo, hi in bands)
    g["those with a lead time of 21 to 26 periods: world (lead, slip)"] = ", ".join(
        f"{s} ({worlds[s].lead}, {worlds[s].slip})" for s in later if 21 <= worlds[s].lead <= 26)
    g["in every world in which it begins later, overloads of the early window; of C6: value (worlds)"] = values(count(
        f"{len(runs[('early-window', s)]['overloads'])}; {len(runs[('C6', s)]['overloads'])}"
        for s in later))
    g["in every world in which it begins later, the delay of the early window less C6's, in periods: value (worlds)"] = (
        values(count(runs[("early-window", s)]["delay"] - runs[("C6", s)]["delay"]
                     for s in later)))
    facts["the early window's late runs"] = g

    # The worlds that are not valid
    g = {}
    for s in sorted(w for w in worlds if w not in valid):
        w = worlds[s]
        parts = []
        for k in co.KINDS:
            m = runs[(k, s)]
            parts.append(f"{k}: {' '.join(map(str, m['alarms'])) or 'none'}, "
                         f"{' '.join(map(str, m['restarts'])) or 'none'}, "
                         f"{'never' if m['begun'] is None else m['begun']}, "
                         f"{float(m['least_clearance']):.4f}")
        g[f"world {s} (survey error {float(w.survey_error):+.2f} m): the alarm; the restart; "
          f"the operation begun; the least true clearance [m]"] = "; ".join(parts)
    below = {k: sorted(s for s in worlds if s not in valid
                       and runs[(k, s)]["least_clearance"] < ce.EPS_MIN) for k in co.KINDS}
    g["worlds whose least true clearance fell below the floor, by configuration; "
      "runs with a violation"] = (
        "; ".join(f"{k}: {', '.join(map(str, below[k])) or 'none'}" for k in co.KINDS)
        + f"; {sum(1 for k in co.KINDS for s in worlds if runs[(k, s)]['violation'])}")
    facts["the worlds that are not valid"] = g

    # Figure S17.2: the operation's delay against the lead time, every valid world
    with open(os.path.join(HERE, "results", "contribution-leads.csv"), "w",
              encoding="utf-8") as fh:
        fh.write("world,lead,slip,delay_c6,delay_fr,delay_ew\n")
        for s in valid:
            d = [runs[(k, s)]["delay"] for k in ("C6", "full-reverse", "early-window")]
            fh.write(f"{s},{worlds[s].lead},{worlds[s].slip},"
                     + ",".join("" if x is None else str(x) for x in d) + "\n")

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
    out_dir = os.path.join(HERE, "results")
    with open(os.path.join(out_dir, "contribution-diagnosis.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if failed:
        raise SystemExit(1)


EXPECTED = {   # every value of the summary but the versions of the libraries
 'F2, the full reverse in the family': {'valid worlds with a lead time of 19 or 20 periods: world (slip)': '16179 '
                                                                                                           '(-6)',
                                        "of these, without a slip (F2's subset in the family)": '0',
                                        'world 16179, the full reverse begins earlier than C6: lead; slip; the forecast onset; the true onset; begun, the full reverse and C6; overloads, C6': '20; '
                                                                                                                                                                                               '-6; '
                                                                                                                                                                                               '216; '
                                                                                                                                                                                               '210; '
                                                                                                                                                                                               '216, '
                                                                                                                                                                                               '237; '
                                                                                                                                                                                               '210 '
                                                                                                                                                                                               '216'},
 'F4, the full reverse in the worlds without a slip and a lead time of 21 periods or more': {"worlds of F4's subset": '58',
                                                                                             'the period in which the governor first commits an ambition below 1 after the announcement, the full reverse less C6: value (worlds)': '0 '
                                                                                                                                                                                                                                    '(58)',
                                                                                             'the period in which the cart first moves backwards after the announcement, the full reverse less C6: value (worlds)': '0 '
                                                                                                                                                                                                                    '(58)',
                                                                                             'worlds in which the governor commits the full reverse; periods in which it does, least and most': '58; '
                                                                                                                                                                                                '14 '
                                                                                                                                                                                                'and '
                                                                                                                                                                                                '14',
                                                                                             'its first commitment, less the period in which C6 first commits an ambition below 1: value (worlds)': '1 '
                                                                                                                                                                                                    '(58)',
                                                                                             "worlds in which an executed command differs from C6's before the operation begins": '0'},
 'E4, the early window in the worlds without a slip and a lead time of 21 to 30 periods': {"worlds of E4's subset": '11',
                                                                                           "world 16101: lead; the early window's onset; the cart first at rest after the announcement, period and place [m]; begun, the early window and C6": '27; '
                                                                                                                                                                                                                                               '208; '
                                                                                                                                                                                                                                               '212, '
                                                                                                                                                                                                                                               '23.6800; '
                                                                                                                                                                                                                                               '218, '
                                                                                                                                                                                                                                               '218',
                                                                                           "world 16102: lead; the early window's onset; the cart first at rest after the announcement, period and place [m]; begun, the early window and C6": '28; '
                                                                                                                                                                                                                                               '225; '
                                                                                                                                                                                                                                               '228, '
                                                                                                                                                                                                                                               '23.6962; '
                                                                                                                                                                                                                                               '235, '
                                                                                                                                                                                                                                               '235',
                                                                                           "world 16111: lead; the early window's onset; the cart first at rest after the announcement, period and place [m]; begun, the early window and C6": '24; '
                                                                                                                                                                                                                                               '214; '
                                                                                                                                                                                                                                               '217, '
                                                                                                                                                                                                                                               '23.9913; '
                                                                                                                                                                                                                                               '266, '
                                                                                                                                                                                                                                               '224',
                                                                                           "world 16117: lead; the early window's onset; the cart first at rest after the announcement, period and place [m]; begun, the early window and C6": '22; '
                                                                                                                                                                                                                                               '195; '
                                                                                                                                                                                                                                               '198, '
                                                                                                                                                                                                                                               '24.1363; '
                                                                                                                                                                                                                                               '247, '
                                                                                                                                                                                                                                               '205',
                                                                                           "world 16118: lead; the early window's onset; the cart first at rest after the announcement, period and place [m]; begun, the early window and C6": '22; '
                                                                                                                                                                                                                                               '198; '
                                                                                                                                                                                                                                               '201, '
                                                                                                                                                                                                                                               '24.1537; '
                                                                                                                                                                                                                                               '250, '
                                                                                                                                                                                                                                               '208',
                                                                                           "world 16122: lead; the early window's onset; the cart first at rest after the announcement, period and place [m]; begun, the early window and C6": '26; '
                                                                                                                                                                                                                                               '222; '
                                                                                                                                                                                                                                               '226, '
                                                                                                                                                                                                                                               '23.8412; '
                                                                                                                                                                                                                                               '253, '
                                                                                                                                                                                                                                               '232',
                                                                                           "world 16128: lead; the early window's onset; the cart first at rest after the announcement, period and place [m]; begun, the early window and C6": '27; '
                                                                                                                                                                                                                                               '203; '
                                                                                                                                                                                                                                               '207, '
                                                                                                                                                                                                                                               '23.6988; '
                                                                                                                                                                                                                                               '213, '
                                                                                                                                                                                                                                               '213',
                                                                                           "world 16140: lead; the early window's onset; the cart first at rest after the announcement, period and place [m]; begun, the early window and C6": '30; '
                                                                                                                                                                                                                                               '195; '
                                                                                                                                                                                                                                               '196, '
                                                                                                                                                                                                                                               '23.6975; '
                                                                                                                                                                                                                                               '205, '
                                                                                                                                                                                                                                               '205',
                                                                                           "world 16150: lead; the early window's onset; the cart first at rest after the announcement, period and place [m]; begun, the early window and C6": '22; '
                                                                                                                                                                                                                                               '203; '
                                                                                                                                                                                                                                               '206, '
                                                                                                                                                                                                                                               '24.1525; '
                                                                                                                                                                                                                                               '255, '
                                                                                                                                                                                                                                               '213',
                                                                                           "world 16180: lead; the early window's onset; the cart first at rest after the announcement, period and place [m]; begun, the early window and C6": '23; '
                                                                                                                                                                                                                                               '229; '
                                                                                                                                                                                                                                               '232, '
                                                                                                                                                                                                                                               '24.0300; '
                                                                                                                                                                                                                                               '281, '
                                                                                                                                                                                                                                               '239',
                                                                                           "world 16184: lead; the early window's onset; the cart first at rest after the announcement, period and place [m]; begun, the early window and C6": '26; '
                                                                                                                                                                                                                                               '218; '
                                                                                                                                                                                                                                               '222, '
                                                                                                                                                                                                                                               '23.7650; '
                                                                                                                                                                                                                                               '249, '
                                                                                                                                                                                                                                               '228',
                                                                                           'first at rest at or before 23.7 m: worlds; their lead times; on time': '4; '
                                                                                                                                                                   '27, '
                                                                                                                                                                   '28, '
                                                                                                                                                                   '27, '
                                                                                                                                                                   '30; '
                                                                                                                                                                   '4',
                                                                                           'first at rest beyond 23.7 m: worlds; their lead times; on time': '7; '
                                                                                                                                                             '24, '
                                                                                                                                                             '22, '
                                                                                                                                                             '22, '
                                                                                                                                                             '26, '
                                                                                                                                                             '22, '
                                                                                                                                                             '23, '
                                                                                                                                                             '26; '
                                                                                                                                                             '0'},
 "the early window's late runs": {'valid worlds in which the early window begins the operation later than C6, by lead time: 20 or fewer; 21 to 26; 27 or more': '15; '
                                                                                                                                                                '9; '
                                                                                                                                                                '0',
                                  'of these, C6 on time': '0; 9; 0',
                                  'those with a lead time of 21 to 26 periods: world (lead, slip)': '16103 '
                                                                                                    '(22, '
                                                                                                    '2), '
                                                                                                    '16111 '
                                                                                                    '(24, '
                                                                                                    '0), '
                                                                                                    '16117 '
                                                                                                    '(22, '
                                                                                                    '0), '
                                                                                                    '16118 '
                                                                                                    '(22, '
                                                                                                    '0), '
                                                                                                    '16122 '
                                                                                                    '(26, '
                                                                                                    '0), '
                                                                                                    '16150 '
                                                                                                    '(22, '
                                                                                                    '0), '
                                                                                                    '16174 '
                                                                                                    '(21, '
                                                                                                    '1), '
                                                                                                    '16180 '
                                                                                                    '(23, '
                                                                                                    '0), '
                                                                                                    '16184 '
                                                                                                    '(26, '
                                                                                                    '0)',
                                  'in every world in which it begins later, overloads of the early window; of C6: value (worlds)': '1; '
                                                                                                                                   '0 '
                                                                                                                                   '(2), '
                                                                                                                                   '2; '
                                                                                                                                   '0 '
                                                                                                                                   '(7), '
                                                                                                                                   '3; '
                                                                                                                                   '1 '
                                                                                                                                   '(13), '
                                                                                                                                   '3; '
                                                                                                                                   '2 '
                                                                                                                                   '(1), '
                                                                                                                                   '4; '
                                                                                                                                   '2 '
                                                                                                                                   '(1)',
                                  "in every world in which it begins later, the delay of the early window less C6's, in periods: value (worlds)": '21 '
                                                                                                                                                  '(3), '
                                                                                                                                                  '42 '
                                                                                                                                                  '(21)'},
 'the worlds that are not valid': {'world 16107 (survey error +0.47 m): the alarm; the restart; the operation begun; the least true clearance [m]': 'C6: '
                                                                                                                                                    '58, '
                                                                                                                                                    '98, '
                                                                                                                                                    '230, '
                                                                                                                                                    '0.5100; '
                                                                                                                                                    'full-reverse: '
                                                                                                                                                    '58, '
                                                                                                                                                    '98, '
                                                                                                                                                    '230, '
                                                                                                                                                    '0.5100; '
                                                                                                                                                    'early-window: '
                                                                                                                                                    '58, '
                                                                                                                                                    '98, '
                                                                                                                                                    '272, '
                                                                                                                                                    '0.5100',
                                   'world 16116 (survey error +0.50 m): the alarm; the restart; the operation begun; the least true clearance [m]': 'C6: '
                                                                                                                                                    '50, '
                                                                                                                                                    'none, '
                                                                                                                                                    'never, '
                                                                                                                                                    '0.4800; '
                                                                                                                                                    'full-reverse: '
                                                                                                                                                    '50, '
                                                                                                                                                    'none, '
                                                                                                                                                    'never, '
                                                                                                                                                    '0.4800; '
                                                                                                                                                    'early-window: '
                                                                                                                                                    '50, '
                                                                                                                                                    'none, '
                                                                                                                                                    'never, '
                                                                                                                                                    '0.4800',
                                   'world 16120 (survey error +0.44 m): the alarm; the restart; the operation begun; the least true clearance [m]': 'C6: '
                                                                                                                                                    '58, '
                                                                                                                                                    'none, '
                                                                                                                                                    'never, '
                                                                                                                                                    '0.4800; '
                                                                                                                                                    'full-reverse: '
                                                                                                                                                    '58, '
                                                                                                                                                    'none, '
                                                                                                                                                    'never, '
                                                                                                                                                    '0.4800; '
                                                                                                                                                    'early-window: '
                                                                                                                                                    '58, '
                                                                                                                                                    'none, '
                                                                                                                                                    'never, '
                                                                                                                                                    '0.4800',
                                   'world 16144 (survey error +0.59 m): the alarm; the restart; the operation begun; the least true clearance [m]': 'C6: '
                                                                                                                                                    '56, '
                                                                                                                                                    'none, '
                                                                                                                                                    'never, '
                                                                                                                                                    '0.4900; '
                                                                                                                                                    'full-reverse: '
                                                                                                                                                    '56, '
                                                                                                                                                    'none, '
                                                                                                                                                    'never, '
                                                                                                                                                    '0.4900; '
                                                                                                                                                    'early-window: '
                                                                                                                                                    '56, '
                                                                                                                                                    'none, '
                                                                                                                                                    'never, '
                                                                                                                                                    '0.4900',
                                   'world 16197 (survey error +0.57 m): the alarm; the restart; the operation begun; the least true clearance [m]': 'C6: '
                                                                                                                                                    '53, '
                                                                                                                                                    '103, '
                                                                                                                                                    '222, '
                                                                                                                                                    '0.5281; '
                                                                                                                                                    'full-reverse: '
                                                                                                                                                    '53, '
                                                                                                                                                    '103, '
                                                                                                                                                    '222, '
                                                                                                                                                    '0.5281; '
                                                                                                                                                    'early-window: '
                                                                                                                                                    '53, '
                                                                                                                                                    '103, '
                                                                                                                                                    '222, '
                                                                                                                                                    '0.5350',
                                   'worlds whose least true clearance fell below the floor, by configuration; runs with a violation': 'C6: '
                                                                                                                                      '16116, '
                                                                                                                                      '16120, '
                                                                                                                                      '16144; '
                                                                                                                                      'full-reverse: '
                                                                                                                                      '16116, '
                                                                                                                                      '16120, '
                                                                                                                                      '16144; '
                                                                                                                                      'early-window: '
                                                                                                                                      '16116, '
                                                                                                                                      '16120, '
                                                                                                                                      '16144; '
                                                                                                                                      '0'}}


if __name__ == "__main__":
    main()
