"""The cart of Chapter 3 in Genesis (Section 12.1).

Genesis reads the MJCF text of Section 11.1 through its own MJCF morph: the
same five links (the world, which carries the ground and the wall, the
chassis, two wheels and the tool), the same six joints, and the actuators
of the text, the wheels' motors limited to 40 N m and the tool's position
servo with gains 1000 N/m and 20 N s/m, limited to 100 N. Genesis ignores
the simulator options of an MJCF text; they are set on the scene instead.
The settings offered:

  "mujoco"      the options of the MJCF text (implicitfast, Newton,
                elliptic cone, 100 iterations, tolerance 1e-8), Genesis's
                MuJoCo compatibility, MuJoCo's default contact time constant
                of 0.02 s, double precision, no decimation of the meshes,
                and GJK collision detection
  "mujoco-mpr"  the same with Genesis's default collision detection, MPR,
                under which a wheel sinks into the crest of the ground
  "mujoco-f32"  the same as "mujoco" in single precision (Section 12.2)
  "defaults"    Genesis's own defaults: single precision, its approximate
                implicitfast integrator, a pyramidal cone, 25 iterations,
                contact pruning, a contact time constant of 0.01 s, MPR and
                no compatibility with MuJoCo
                (the ground's mesh, a convex hull of 680 faces, is kept as
                written either way; the record reports it)

Genesis computes on the CPU here (backend gs.cpu). The precision is chosen
when Genesis is initialized, once per process, so a process runs one
setting. The wheel torques go through control_dofs_force, the tool's servo
through control_dofs_position; the wheels' loads are their net contact
forces along the ground's normal. The state is read from the joint
coordinates (see _kinematics). camera=(width, height) adds a camera,
rendered by Genesis's rasterizer when render() is called (Section 12.2); it
looks at the chassis from the side.
"""
import math

import numpy as np

import genesis as gs

from .base import Kin, PortedCart
from .design import mjcf

SETTINGS = ("mujoco", "mujoco-mpr", "mujoco-f32", "defaults")
_INITIALIZED = None


def init(precision):
    """Initializes Genesis on the CPU, once per process."""
    global _INITIALIZED
    if _INITIALIZED is None:
        gs.init(backend=gs.cpu, precision=precision, logging_level="warning")
        _INITIALIZED = precision
    elif _INITIALIZED != precision:
        raise RuntimeError(f"Genesis was initialized with precision {_INITIALIZED}; "
                           f"{precision} needs a new process")


def _np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


class GenesisCart(PortedCart):
    """The cart in Genesis; see the module docstring."""
    BACKEND = "genesis"

    def __init__(self, design, settings="mujoco", timestep=None, coordinate="axle",
                 camera=None):
        if settings not in SETTINGS:
            raise ValueError(f"unknown settings {settings!r}; one of {SETTINGS}")
        self.settings = settings
        self.camera_res, self.camera, self.frames = camera, None, 0
        super().__init__(design, timestep, coordinate)

    # --- the model and the solver ---------------------------------------------------
    def _build(self):
        d = self.design
        if self.settings in ("mujoco", "mujoco-mpr", "mujoco-f32"):
            init("32" if self.settings == "mujoco-f32" else "64")
            rigid = gs.options.RigidOptions(
                integrator=gs.integrator.implicitfast,
                constraint_solver=gs.constraint_solver.Newton,
                iterations=d.iterations, tolerance=d.tolerance,
                friction_cone=gs.friction_cone.elliptic,
                enable_mujoco_compatibility=True, constraint_timeconst=0.02,
                use_gjk_collision=(self.settings != "mujoco-mpr"))
            morph = gs.morphs.MJCF(file=mjcf(d), decimate=False)
        else:
            init("32")
            rigid = gs.options.RigidOptions()
            morph = gs.morphs.MJCF(file=mjcf(d))
        extra = {} if self.camera_res is None else {"renderer": gs.renderers.Rasterizer()}
        self.scene = gs.Scene(sim_options=gs.options.SimOptions(dt=self.dt, substeps=1),
                              rigid_options=rigid, show_viewer=False, **extra)
        self.entity = self.scene.add_entity(morph)
        if self.camera_res is not None:
            self.camera = self.scene.add_camera(res=tuple(self.camera_res), pos=(1.5, -2.5, 1.2),
                                                lookat=(0.0, 0.0, 0.2), fov=40, GUI=False)
        self.scene.build()
        names = [link.name for link in self.entity.links]
        self.link = {n: names.index(n) for n in ("chassis", "front", "rear", "tool")}
        joints = [j.name for j in self.entity.joints]
        if joints != ["x", "z", "pitch", "front", "rear", "reach"]:
            raise RuntimeError(f"unexpected joints {joints}")

    # --- the operations of a port ------------------------------------------------------
    def _kinematics(self):
        """From the joint coordinates, which get_qpos returns as the step left
        them. Under MuJoCo compatibility, Genesis updates the link poses and
        the DOF positions that the entity reports at the start of the next
        step, as MuJoCo does its derived quantities: read after a step, they
        are a step old (get_dofs_position lags get_qpos by one step). The axles
        are placed on the chassis as in the MJCF text."""
        e, d = self.entity, self.design
        q, qd = _np(e.get_qpos()), _np(e.get_dofs_velocity())
        x, z, th = float(q[0]), float(q[1]), float(q[2])
        vx, vz, w = float(qd[0]), float(qd[1]), float(qd[2])
        cx, cz = -d.half_length + x, d.wheel_radius + d.wheel_drop + z
        c, s = math.cos(th), math.sin(th)

        def axle(ox, oz):              # an axle at (ox, oz) on the chassis
            rx, rz = ox * c + oz * s, -ox * s + oz * c
            return (cx + rx, cz + rz), (vx + w * rz, vz - w * rx)

        front, front_vel = axle(d.wheel_offset, -d.wheel_drop)
        rear, _ = axle(-d.wheel_offset, -d.wheel_drop)
        return Kin(front=front, front_vel=front_vel, rear=rear, pitch=th, reach=float(q[5]),
                   front_rate=float(qd[3]), rear_rate=float(qd[4]))

    def _contact_loads(self):
        forces = _np(self.entity.get_links_net_contact_force())
        out = []
        for w in ("front", "rear"):
            f = forces[self.link[w]]
            nx, nz = self.pr.normal(self.foot(w).s)
            out.append(abs(float(f[0] * nx + f[2] * nz)))
        return tuple(out)

    def _write(self, q, qd):
        e = self.entity
        e.set_dofs_position(np.asarray(q, dtype=float), zero_velocity=False)
        e.set_dofs_velocity(np.asarray(qd, dtype=float))

    def _apply(self, torque_front, torque_rear, ell):
        e = self.entity
        e.control_dofs_force(np.array([torque_front, torque_rear]), [3, 4])
        e.control_dofs_position(np.array([ell]), [5])

    def _step(self):
        self.scene.step()

    def render(self):
        """An image of the camera, which looks at the chassis from the side; the
        pose of the camera is set from the joint coordinates, which it only
        reads."""
        x = float(_np(self.entity.get_qpos())[0]) - self.design.half_length
        self.camera.set_pose(pos=(x + 1.5, -2.5, 1.2), lookat=(x, 0.0, 0.2))
        self.frames += 1
        return self.camera.render()[0]

    # --- the record -----------------------------------------------------------------------
    def record(self):
        e, rs = self.entity, self.scene.sim.rigid_solver
        # what the entity reports after a step, against the joint coordinates
        self.place(5.0, 1.0, 0.0)
        self.controls(0.0, 1.0, 0.0)
        self.step()
        x = float(_np(e.get_qpos())[0])
        lag = float(_np(e.get_dofs_position())[0]) - x
        link_lag = (float(_np(e.get_links_pos())[self.link["chassis"]][0])
                    - (x - self.design.half_length))
        o = rs._options
        ground = e.geoms[0]
        lo, hi = (_np(x) for x in e.get_dofs_force_range())
        bias = [_np(b) for b in e.get_dofs_act_bias()]
        return {
            "settings": self.settings,
            "precision": _INITIALIZED,
            "links, joints, degrees of freedom": f"{e.n_links}, {len(e.joints)}, {e.n_dofs}",
            "link masses [kg]": ", ".join(f"{float(link.get_mass()):.6g}" for link in e.links),
            "rigid options: integrator, constraint solver, cone, iterations, tolerance":
                f"{o.integrator.name}, {o.constraint_solver.name}, {o.friction_cone.name}, "
                f"{o.iterations}, {o.tolerance:g}",
            "rigid options: MuJoCo compatibility, impratio, contact time constant [s], "
            "contact pruning, contact resolution, GJK":
                f"{o.enable_mujoco_compatibility}, {o.impratio}, {o.constraint_timeconst}, "
                f"{o.contact_pruning_tolerance}, "
                f"{o.contact_resolution.name if o.contact_resolution is not None else None}, "
                f"{o.use_gjk_collision}",
            "ground mesh: vertices, faces": f"{ground.n_verts}, {ground.n_faces}",
            "friction of the ground and the wheels": ", ".join(
                f"{float(e.geoms[i].get_friction()):g}" for i in (0, 3, 4)),
            "actuators: wheel force range [N m], tool gain, bias, force range": (
                f"{lo[3]:g} to {hi[3]:g}; {float(_np(e.get_dofs_act_gain())[5]):g}, "
                f"{bias[1][5]:g} {bias[2][5]:g}, {lo[5]:g} to {hi[5]:g}"),
            "state precision": "float64" if _INITIALIZED == "64" else "float32",
            "one step at 1 m/s from 5 m: the entity's DOF position x less the joint "
            "coordinate x [m]": f"{lag:.3g}",
            "one step at 1 m/s from 5 m: the chassis link's position x less that of the "
            "joint coordinates [m]": f"{link_lag:.3g}",
        }
