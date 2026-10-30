"""
The repository's mission with one robot.

Physical AI, Together, Chapter 15 (Section 15.3).

The Team-Impedance-Control repository (Heydari Shahna, 2026) simulates two
to four Apptronik Apollo humanoids that carry a girder together with a
two-loop controller (teamimp/controller.py): outer loops of an external
impedance for the payload and an internal impedance for the forces that
share its load, whose references a QP chooses with penalized slacks on the
contacts' wrench cones and the actuators' torques (teamimp/share.py), and
an inner loop from a fitted reference to the efforts. Its package teamimp
is loaded here from a checkout, unchanged; what one robot needs that the
package does not have is added from outside it:

  * a station for one robot (scene.Layout knows two, three and four): on the
    -y side of the beam, facing +y, so that it walks forwards both to the
    beam and when it carries it;
  * the beam: an aluminium I-beam with the cross-section of an HEA 200
    (EN 10365: h 190 mm, b 200 mm, web 6.5 mm, flanges 10 mm; 42.3 kg/m in
    steel), its mass scaled by the densities of aluminium and steel, 2700
    and 7850 kg/m^3, as the repository scales its "AL 280" girder;
  * in a team, the other robots take up the load while one shifts its
    weight onto its stance foot, so the repository shifts a loaded robot's
    pelvis only a quarter of the way over that foot (shift_frac_load); one
    robot must shift its weight, load included, as far as it does unloaded
    (0.95, the repository's shift_frac_free), and the beam's reference
    follows the pelvis's reference sideways while it is carried, so that
    the arms need not stretch across the shift: without this
    (humanoid_carry.py --without-follow, adapt(mods, follow=False)), the
    wrists' roll actuators reach their torque limits in the first loaded
    step and the shift after it, 11 of the legs' 12 actuators reach theirs,
    and the girder is lost in the second loaded step.

Everything else, the controller, its gains, the load sharing, the gait and
the mission's phases, is the repository's.
"""
import os
import sys

import numpy as np

PROFILE = "AL 200"
HEA_200 = (0.190, 0.200, 0.0065, 0.010, 42.3)    # h, b, tw, tf [m]; mass per length [kg/m], steel
ALUMINIUM, STEEL = 2700.0, 7850.0                 # [kg/m^3], the repository's densities
LENGTH = 1.2                                      # [m] the beam (ours)
TRESTLES = 0.5                                    # [m] the trestles' x, either side of the middle
APPROACH, CARRY, RETREAT = 2, 4, 1                # gait cycles of 0.25 m
STATION = (0.0, -1)                               # x along the beam, side: the robot on -y


def load(checkout, qp="any"):
    """Imports teamimp from the checkout and returns its modules. The
    package's share.py imports daqp; qp = "nnls" puts humanoid/qp.py in its
    place, "daqp" needs daqp, "any" (for drawing, which solves no QP) takes
    daqp if it is installed and qp.py otherwise."""
    checkout = os.path.abspath(checkout)
    if not os.path.isdir(os.path.join(checkout, "teamimp")):
        raise SystemExit(f"{checkout} is not a checkout of Team-Impedance-Control (no teamimp/)")
    if qp == "nnls" or (qp == "any" and "daqp" not in sys.modules):
        try:
            if qp == "nnls":
                raise ImportError
            import daqp                                   # noqa: F401
        except ImportError:
            from . import qp as solver
            sys.modules["daqp"] = solver                  # share.py calls daqp.solve
    sys.path.insert(0, checkout)
    from teamimp import controller, mission, scene, share, sim_mj, team
    return dict(controller=controller, mission=mission, scene=scene, share=share, sim_mj=sim_mj,
                team=team)


def commit(checkout):
    """The checkout's commit, read from its .git folder, or None."""
    git = os.path.join(checkout, ".git")
    try:
        with open(os.path.join(git, "HEAD"), encoding="utf-8") as fh:
            head = fh.read().strip()
        if not head.startswith("ref:"):
            return head
        ref = head.split(" ", 1)[1]
        path = os.path.join(git, ref)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                return fh.read().strip()
        with open(os.path.join(git, "packed-refs"), encoding="utf-8") as fh:
            for line in fh:
                if line.strip().endswith(ref):
                    return line.split()[0]
    except OSError:
        return None
    return None


def adapt(mods, follow=True):
    """Adds one robot's station and the beam's profile to the repository's
    scene, and returns the configuration and the mission's class for one
    robot. follow=False leaves the beam's reference where the repository
    puts it, without the pelvis's sideways shift (for the run that shows why
    one robot needs it)."""
    scene, mission = mods["scene"], mods["mission"]
    original = scene.Layout.__init__

    def layout(self, n, x_sp=0.8, y_off=0.58):
        if n == 1:
            self.n, self.stations, self.y_off = 1, [STATION], y_off
        else:
            original(self, n, x_sp, y_off)

    if not getattr(scene.Layout, "_one_robot", False):
        scene.Layout.__init__ = layout
        scene.Layout._one_robot = True
    h, b, tw, tf, mpl = HEA_200
    scene.PROFILES[PROFILE] = (h, b, tw, tf, mpl * ALUMINIUM / STEEL)

    class Refs(mission.MissionRefs):
        """The beam's reference, shifted sideways with the pelvis's reference
        while the beam is carried."""

        def __init__(self):
            super().__init__()
            self.follow = None

        def payload(self, t):
            p, R, v, w, a, wd = self.pay(t)
            if self.follow is None:
                return p, R, v, w, a, wd
            e, x0 = self.follow
            pp, _, pv, _, pa, _ = self.pel[0](t)
            return (p + ((pp @ e) - x0) * e, R, v + (pv @ e) * e, w, a + (pa @ e) * e, wd)

    class Mission(mission.Mission):
        def _init_state(self):
            super()._init_state()
            refs = Refs()
            refs.__dict__.update(self.refs.__dict__)
            refs.follow = None
            self.refs = refs
            self.C.refs = refs

        def sideways(self):
            return mission.rot_z(-self.side[0] * np.pi / 2)[:, 1]

        def close_grasps(self):
            super().close_grasps()
            if follow:
                e = self.sideways()
                self.refs.follow = (e, float(self.refs.pel[0](self.R.t)[0] @ e))

        def open_grasps(self):
            self.refs.follow = None
            super().open_grasps()

        def _log(self):
            super()._log()
            if self.tm.mode.payload:
                self.logs[-1]["beam_ref"] = self.refs.payload(self.R.t)[0].copy()

    cfg = mission.Cfg(n=1, profile=PROFILE, beam_len=LENGTH, x_sup=TRESTLES, kz=0.0,
                      shift_frac_load=0.95, D_app=APPROACH * 0.25, YB=CARRY * 0.25, t_max=200.0)
    return cfg, Mission


def phases(mods, M):
    return mods["mission"].full_mission(M, n_app=APPROACH, n_carry=CARRY, retreat=RETREAT)
