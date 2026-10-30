"""The decision rules of the anticipation benchmark's hypotheses, as
HYPOTHESES.md states them (Section 16.1). Each rule returns whether the
hypothesis is supported, how many cases it examined, and every exception,
and, for the report, the same for each of its statements (added after the
first full run, as HYPOTHESES.md records; the rules and their verdicts are
unchanged).

runs[(config, seed)] is the evaluator's dictionary of a run; worlds[seed]
the world; valid the seeds of the valid worlds.
"""
from cart.model import ce

FLOOR = ce.EPS_MIN
WORK_ZONE = ce.WORK_ZONE


class Tally:
    """Cases and exceptions of a hypothesis, by statement. The verdict is
    the rule of HYPOTHESES.md: supported if no statement has an exception;
    its cases and exceptions are those of all its statements, in the order
    in which the rule examined them."""

    def __init__(self, *statements):
        self.statements = statements             # (tag, description)
        self.cases = {tag: 0 for tag, _ in statements}
        self.exc = []                            # (tag, world, message)

    def case(self, tag, n=1):
        self.cases[tag] += n

    def fail(self, tag, world, message):
        self.exc.append((tag, world, message))

    def verdict(self):
        exceptions = [m for _, _, m in self.exc]
        return dict(
            supported=not exceptions, cases=sum(self.cases.values()), exceptions=exceptions,
            statements=[(tag, text, self.cases[tag],
                         [w for t, w, _ in self.exc if t == tag]) for tag, text in self.statements])


def a1(runs):
    t = Tally(("1", "no alarm and a true clearance of at least the floor in every run whose "
                    "executed labels are valid"))
    for (cid, seed), m in sorted(runs.items()):
        if m["invalid"]:
            continue
        t.case("1")
        if m["alarms"] or m["least_clearance"] < FLOOR:
            t.fail("1", seed, f"{cid} world {seed}: alarms {m['alarms']}, least clearance "
                              f"{float(m['least_clearance']):.3f} m")
    return t.verdict()


def a2(runs, worlds, valid):
    t = Tally(("1", "C1 to C3 never begin the operation when the cart is in the work zone at "
                    "the onset"))
    for cid in ("C1", "C2", "C3"):
        for seed in sorted(valid):
            m = runs[(cid, seed)]
            if m["p_onset"] is None or m["p_onset"] < WORK_ZONE:
                continue
            t.case("1")
            if m["begun"] is not None:
                t.fail("1", seed, f"{cid} world {seed}: begun in period {m['begun']}")
    return t.verdict()


def a3(runs, worlds, valid):
    t = Tally(("1", "no slip, a lead time of 21 periods or more: C6, C7 and C8 on time"),
              ("2", "no slip, a lead time of 18 periods or fewer: C6, C7 and C8 not on time"),
              ("3", "no slip: C5 21 periods late when the cart is in the work zone at the onset"))
    for seed in sorted(valid):
        w = worlds[seed]
        if w.slip != 0:
            continue
        for cid in ("C6", "C7", "C8"):
            d = runs[(cid, seed)]["delay"]
            if w.lead >= 21:
                t.case("1")
                if d != 0:
                    t.fail("1", seed, f"{cid} world {seed} (lead {w.lead}): delay {d}")
            elif w.lead <= 18:
                t.case("2")
                if d == 0:
                    t.fail("2", seed, f"{cid} world {seed} (lead {w.lead}): on time")
        m = runs[("C5", seed)]
        if m["p_onset"] is not None and m["p_onset"] >= WORK_ZONE:
            t.case("3")
            if m["delay"] != 21:
                t.fail("3", seed, f"C5 world {seed}: delay {m['delay']}")
    return t.verdict()


def a4(runs, worlds, valid):
    t = Tally(("1", "C4 on time and never in the work zone"))
    for seed in sorted(valid):
        m = runs[("C4", seed)]
        t.case("1")
        if m["delay"] != 0 or m["arrival"] is not None:
            t.fail("1", seed, f"C4 world {seed}: delay {m['delay']}, arrival {m['arrival']}")
    return t.verdict()


def _lt(a, b):
    """a < b for arrival periods, None meaning never."""
    if a is None:
        return False
    return b is None or a < b


def a5(runs, worlds, valid):
    t = Tally(("1", "C6 arrives before C7"), ("2", "C8 arrives no later than C6"),
              ("3", "C6 and C7 begin the operation in the same period"),
              ("4", "every world: C2 arrives in the same period as C1"))
    for seed in sorted(valid):
        r6, r7, r8 = (runs[(c, seed)] for c in ("C6", "C7", "C8"))
        for tag in ("1", "2", "3"):
            t.case(tag)
        if not _lt(r6["arrival"], r7["arrival"]):
            t.fail("1", seed, f"world {seed}: arrival C6 {r6['arrival']}, C7 {r7['arrival']}")
        if _lt(r6["arrival"], r8["arrival"]):
            t.fail("2", seed, f"world {seed}: arrival C8 {r8['arrival']} after C6 "
                              f"{r6['arrival']}")
        if r6["begun"] != r7["begun"]:
            t.fail("3", seed, f"world {seed}: begun C6 {r6['begun']}, C7 {r7['begun']}")
    for seed in sorted(worlds):
        t.case("4")
        if runs[("C1", seed)]["arrival"] != runs[("C2", seed)]["arrival"]:
            t.fail("4", seed, f"world {seed}: arrival C1 {runs[('C1', seed)]['arrival']}, "
                              f"C2 {runs[('C2', seed)]['arrival']}")
    return t.verdict()


A6_KEYS = ("begun", "arrival", "in_work_zone")


def a6(runs, worlds, valid):
    t = Tally(("1", "C11 as C6: the operation's period, the arrival, the time in the work zone"),
              ("2", "C14 as C11: the operation's period, the arrival, the time in the work zone"),
              ("3", "C13 begins the operation in the same period as C6"),
              ("4", "no slip and a lead time of 27 periods or more: C12 begins later than C6"))
    keys = A6_KEYS
    for seed in sorted(valid):
        w = worlds[seed]
        r6, r11, r12, r13, r14 = (runs[(c, seed)] for c in ("C6", "C11", "C12", "C13", "C14"))
        for tag in ("1", "2", "3"):
            t.case(tag)
        if any(r11[k] != r6[k] for k in keys):
            t.fail("1", seed, f"world {seed}: C11 " + ", ".join(f"{k} {r11[k]}" for k in keys) +
                   "; C6 " + ", ".join(f"{k} {r6[k]}" for k in keys))
        if any(r14[k] != r11[k] for k in keys):
            t.fail("2", seed, f"world {seed}: C14 " + ", ".join(f"{k} {r14[k]}" for k in keys) +
                   "; C11 " + ", ".join(f"{k} {r11[k]}" for k in keys))
        if r13["begun"] != r6["begun"]:
            t.fail("3", seed, f"world {seed}: begun C13 {r13['begun']}, C6 {r6['begun']}")
        if w.slip == 0 and w.lead >= 27:
            t.case("4")
            if not _lt(r6["begun"], r12["begun"]):
                t.fail("4", seed, f"world {seed}: begun C12 {r12['begun']}, C6 {r6['begun']}")
    return t.verdict()


def a6_keys(runs, valid):
    """For the report: the valid worlds in which C11 differs from C6, and C14
    from C11, in each quantity of statements 1 and 2 of A6."""
    out = {}
    for a, b in (("C11", "C6"), ("C14", "C11")):
        for k in A6_KEYS:
            out[(a, b, k)] = [s for s in sorted(valid) if runs[(a, s)][k] != runs[(b, s)][k]]
    return out


def a7(runs, worlds, valid):
    t = Tally(("1", "the true onset 21 periods or more after the announcement: C9 on time"),
              ("2", "a lead time of 21 periods or more and a later onset: C6 on time, 0.1 s "
                    "less in the work zone than C9 per period of slip"),
              ("3", "a lead time of 21 periods or more and an earlier onset: C6 begins at the "
                    "forecast onset"))
    for seed in sorted(valid):
        w = worlds[seed]
        r6, r9 = runs[("C6", seed)], runs[("C9", seed)]
        if w.onset - w.announce >= 21:
            t.case("1")
            if r9["delay"] != 0:
                t.fail("1", seed, f"C9 world {seed}: delay {r9['delay']}")
        if w.lead >= 21 and w.slip > 0:
            t.case("2")
            short = r9["in_work_zone"] - r6["in_work_zone"]
            if r6["delay"] != 0 or short != w.slip:
                t.fail("2", seed, f"world {seed} (slip {w.slip}): C6 delay {r6['delay']}, "
                                  f"periods in the work zone fewer than C9's: {short}")
        if w.lead >= 21 and w.slip < 0:
            t.case("3")
            if r6["begun"] != w.onset_forecast:
                t.fail("3", seed, f"world {seed} (slip {w.slip}): C6 begun {r6['begun']}, "
                                  f"forecast onset {w.onset_forecast}")
    return t.verdict()


def a8(runs, worlds, valid):
    t = Tally(("1", "C15 begins the operation in the same period as C6"))
    for seed in sorted(valid):
        t.case("1")
        if runs[("C15", seed)]["begun"] != runs[("C6", seed)]["begun"]:
            t.fail("1", seed, f"world {seed}: begun C15 {runs[('C15', seed)]['begun']}, "
                              f"C6 {runs[('C6', seed)]['begun']}")
    return t.verdict()


RULES = (("A1", "the certificate", lambda r, w, v: a1(r)),
         ("A2", "no demand channel", a2),
         ("A3", "preparation from the forecast", a3),
         ("A4", "fixed conservatism", a4),
         ("A5", "the capability forecast", a5),
         ("A6", "equal information", a6),
         ("A7", "errors of the forecast", a7),
         ("A8", "the budget", a8))


def decide(runs, worlds, valid):
    return [(hid, name, rule(runs, worlds, valid)) for hid, name, rule in RULES]
