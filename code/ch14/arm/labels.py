"""
The UR5e's labels and reserve: Section 14.2.

Physical AI, Together, Chapter 14 (Section 14.2).

A label of the base joint is a deceleration b, as in Section 14.1. A
predictor that is told the payload issues it from a declaration: an interval
for the payload's mass and one for the distance of its centre of mass from
the flange, along tool0's axis, the payload's box itself (0.1 m, hanging
from the flange) known. The label is valid for the interval if the commanded
stop at b needs, for every payload of the interval, no more than a share of
any joint's torque: the rest of the torque is the feedback's, which Section
14.1 showed has nothing left when the label sits on the bound itself.
interval_label() finds that label on a grid over the interval (its corners,
the middles of its edges and its centre); the bounds are monotone in the
mass and in the distance over these intervals, so that a corner binds. The
global capability's label is the largest deceleration valid for every
payload up to UR's rated 5 kg whose centre of mass lies within the box; as a
label it admits every payload whose stop at it keeps the share, some heavier
than the rating too. Every number here is ours but the rating.

A plan is certified at the payload's clearance from the post at the point of
rest of its recovery, a sound lower bound on its reserve under a norm of the
state that bounds how far a change of it moves any point of the arm and the
payload: stop_angle() and clearance_at() give that point and the clearance
there, the box's place not depending on where its mass sits. The global
capability's label also admits payloads heavier than the rating whose stop
at it needs no more than the share (labels_demand.py finds them).
"""
import math

import mujoco

from . import control, world

FULL = -math.pi                  # [rad/s] the base joint at its speed limit, towards the post
MASS = (4.05, 4.95)              # [kg] the declared payload: 4.5 kg +- 10 % (ours)
OFFSET = (0.025, 0.075)          # [m] its centre of mass from the flange (ours)
GLOBAL_MASS = (0.0, 5.0)         # [kg] the global capability: up to UR's rated payload
GLOBAL_OFFSET = (0.0, 0.10)      # [m] and its centre of mass anywhere in the box
BOX = 0.05                       # [m] the box's centre from the flange: it hangs 0 to 0.1 m below
SHARE = 0.9                      # the share of each joint's torque that a label may use (ours)
FLOOR = 0.01                     # [m] the reserve floor: the least clearance from the post (ours)
GRID = 0.001                     # [m] the reserve's levels: whole millimetres (ours)


def grid(lo_hi, n=3):
    lo, hi = lo_hi
    return [lo + (hi - lo) * k / (n - 1) for k in range(n)]


def arm_for(mass, offset, **kw):
    return control.Arm(payload=mass, payload_offset=offset, box_offset=BOX, **kw)


def interval_label(mass=MASS, offset=OFFSET, share=SHARE, w0=FULL, n=3):
    """The largest deceleration of the base joint whose commanded stop from w0
    needs at most share of every joint's torque for every payload of the grid
    over the interval; with the payload and the joint that bind, and each
    grid point's own bound."""
    best, which, table = math.inf, None, []
    for m in grid(mass, n):
        for r in grid(offset, n):
            b, (joint, end), _ = control.capability(arm_for(m, r), w0, share)
            table.append((m, r, b, joint, end))
            if b < best:
                best, which = b, (m, r, joint, end)
    return best, which, table


def stop_angle(b, start=world.PLACE, w0=FULL):
    """The task's angle at which the commanded stop at b from w0 at start
    comes to rest [rad]."""
    return start - w0 * w0 / (2.0 * b)


def clearance_at(arm, angle):
    """The clearance [m] between the post and the arm with its payload, held
    in the carry pose at the task's angle."""
    d = mujoco.MjData(arm.m)
    d.qpos[:] = world.turned(arm.pose, angle)
    mujoco.mj_kinematics(arm.m, d)
    return world.clearance(arm.m, d)


def latest_brake(arm, b, w0=FULL, floor=FLOOR, tol=1e-7):
    """The latest angle of the swing, towards the post, from which a stop at b
    from w0 still keeps the floor at its point of rest [rad], by bisection
    between the place and the post's centre."""
    ok, bad = world.PLACE, world.post_angle()
    if clearance_at(arm, stop_angle(b, ok, w0)) < floor:
        return None
    while ok - bad > tol:
        mid = 0.5 * (ok + bad)
        ok, bad = (mid, bad) if clearance_at(arm, stop_angle(b, mid, w0)) >= floor else (ok, mid)
    return ok
