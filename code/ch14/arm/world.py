"""
The UR5e's task of Section 14.1: a swing towards a post (ours).

Physical AI, Together, Chapter 14 (Section 14.1).

The arm stands on a table, its base at the origin. It carries a payload in
the carry pose: the tool pointing straight down, tool0 at radius 0.5 m from
the base's axis and 0.25 m above the table. The base joint swings the arm
from the pick to the place; the angles of the task are those of tool0 about
the base's axis, from base_link's x axis (UR's offsets at the shoulder and
the wrists put the base joint's own angle 15.5 degrees behind them in the
carry pose): the pick at +120 degrees, the place at -60 degrees. Beyond the
place, on the side of decreasing angle, stands a post, the cell's fixture: a
box of 0.1 m by 0.1 m by 0.20 m on the table, its centre at radius 0.5 m
and at the angle -60 - POST_GAP degrees. It is lower than the flange, so
that the empty arm passes over it and only a payload below the tool can
reach it. The task's constraints:

  * no contact between the arm or its payload and the post, at any instant;
  * the joints within their ranges, their speeds within 180 deg/s and their
    torques within the limits of joint_limits.yaml (the joints' own values,
    which UR's article says an application cannot fully use);
  * rest at the place.

Every number here is ours, chosen so that the swing's stop is the question.
A rotation about the base's axis, which is vertical, changes neither the
arm's mass matrix nor its gravity, so the carry pose is found once, at the
angle 0, and turned.
"""
import math

import mujoco
import numpy as np

from . import model

CARRY_RADIUS = 0.5          # [m] tool0 from the base's axis in the carry pose
CARRY_HEIGHT = 0.25         # [m] tool0 above the table
PICK = math.radians(120.0)  # [rad] the base joint at the pick
PLACE = math.radians(-60.0)  # [rad] and at the place
POST_GAP = 20.0             # [deg] from the place to the post's centre
POST_SIZE = (0.1, 0.1, 0.20)  # [m] the post's box: lower than the flange, so that only a
                              # payload below the tool reaches it
SEED_POSE = (0.0, -1.414, 2.049, -2.206, -1.571, 0.0)   # starting guess of the carry pose


def post_angle():
    return PLACE - math.radians(POST_GAP)


def add_world(spec):
    """Adds the table (a plane at z = 0, without contacts) and the post."""
    spec.worldbody.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1],
                            contype=0, conaffinity=0, rgba=[0.8, 0.8, 0.75, 1])
    a = post_angle()
    hx, hy, hz = (s / 2 for s in POST_SIZE)
    spec.worldbody.add_geom(name="post", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, hz],
                            pos=[CARRY_RADIUS * math.cos(a), CARRY_RADIUS * math.sin(a), hz],
                            quat=[math.cos(a / 2), 0, 0, math.sin(a / 2)],
                            contype=model.POST_CONTYPE, conaffinity=model.POST_CONAFFINITY,
                            rgba=[0.7, 0.2, 0.2, 1])


def carry_pose(m, d, iterations=400):
    """The carry pose at the base angle 0: tool0 at CARRY_RADIUS and
    CARRY_HEIGHT, its z axis pointing down, found by damped least squares on
    the site's Jacobian. Returns the joint angles and the residual."""
    site = m.site("tool0").id
    target = np.array([CARRY_RADIUS, 0.0, CARRY_HEIGHT])
    q = np.array(SEED_POSE, float)
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    err = np.inf
    for _ in range(iterations):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        z = d.site_xmat[site].reshape(3, 3)[:, 2]
        e = np.concatenate([target - d.site_xpos[site], np.cross(z, [0.0, 0.0, -1.0])])
        err = float(np.linalg.norm(e))
        if err < 1e-12:
            break
        mujoco.mj_jacSite(m, d, jp, jr, site)
        jac = np.vstack([jp, jr])
        q += np.linalg.solve(jac.T @ jac + 1e-9 * np.eye(m.nv), jac.T @ e)
    return q, err


def turned(pose, angle):
    """The carry pose turned about the base's axis so that tool0 lies at the
    given angle of the task (pose[0] is the base joint's angle when tool0
    lies at the angle 0)."""
    q = np.array(pose, float)
    q[0] = pose[0] + angle
    return q


def clearance(m, d, distmax=1.0):
    """The least distance between the post and the arm's geoms, the payload
    included [m]: MuJoCo's distance between the geoms, convex hulls for the
    meshes."""
    post = m.geom("post").id
    best = distmax
    fromto = np.zeros(6)
    for g in range(m.ngeom):
        if g == post or m.geom_contype[g] != model.ARM_CONTYPE:
            continue
        best = min(best, mujoco.mj_geomDistance(m, d, g, post, distmax, fromto))
    return best


def swing_end():
    """The end of the swing that the task's checks search: 10 degrees past
    the post's centre, beyond the post's far side [rad]."""
    return post_angle() - math.radians(10.0)


def least_clearance(m, d, pose, start=None, step=math.radians(0.02)):
    """The least clearance over the swing from the place (or start) to
    swing_end(), by steps of 0.02 degrees, and the angle where it lies
    [m, rad]."""
    a = PLACE if start is None else start
    best = (math.inf, a)
    while a >= swing_end():
        d.qpos[:] = turned(pose, a)
        mujoco.mj_kinematics(m, d)
        best = min(best, (clearance(m, d), a))
        a -= step
    return best


def contact_angle(m, d, pose, start=None, step=1e-3):
    """The angle of the task, swinging from the place towards the post, at
    which the clearance first reaches zero (by stepping, then bisection)
    [rad]; None if the arm passes the post untouched."""
    a = PLACE if start is None else start

    def gap(angle):
        d.qpos[:] = turned(pose, angle)
        mujoco.mj_kinematics(m, d)
        return clearance(m, d)

    lo = a
    while gap(lo - step) > 0:
        lo -= step
        if lo < swing_end():
            return None
    hi = lo - step
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if gap(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
