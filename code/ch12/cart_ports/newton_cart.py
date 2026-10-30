"""The cart of Chapter 3 in Newton (Section 12.1).

Newton builds its model from the MJCF text of Section 11.1 with its own
importer: four bodies, the chassis on one D6 joint that carries the three
joints of MuJoCo's chassis (x, z, pitch), two wheel hinges and the tool's
slide. Newton then simulates the model with one of its solvers:

  "mujoco-warp"     SolverMuJoCo on MuJoCo Warp, MuJoCo's algorithms in Warp
  "mujoco-c"        SolverMuJoCo on MuJoCo's C library (use_mujoco_cpu=True)
  "xpbd"            SolverXPBD, with Newton's collision pipeline
  "featherstone"    SolverFeatherstone, with Newton's collision pipeline
  "semi-implicit"   SolverSemiImplicit, with Newton's collision pipeline

The two MuJoCo solvers rebuild a MuJoCo model from Newton's model, with the
options that the importer read from the MJCF text (integrator, cone, solver,
iterations, tolerance); the time step is set at every step. Featherstone
works in joint coordinates; XPBD and the semi-implicit solver move the
bodies, and the joint coordinates are recovered from them after each step.
The wheel torques go through Newton's generalized joint forces, the tool's
servo through its joint target, which the importer gave the position
actuator's gains, 1000 N/m and 20 N s/m. Newton's state is held in single
precision.

The importer replaces the ground's mesh by its convex hull, as MuJoCo does,
but limits the hull to 64 vertices by default; the ground of Section 11.1
has 342, and a hull of 64 replaces the crest's 166 chords by coarser ones.
The port keeps every vertex (hull_vertices=1000); hull_vertices=64 is the
importer's default, a mistake of conversion that the port checks reveal.
For Newton, a limit of 0 would mean none, but Newton passes the limit on to
MuJoCo, whose convex hull rejects it.

Newton gives every shape a gap of 0.1 m, the distance at which its own
collision pipeline begins to report contacts, and passes it to MuJoCo as
the geom's gap, which in MuJoCo marks a band of inactive contacts beyond the
margin. The MJCF text sets no gap, which in MuJoCo means none; the port sets
the gap to 0 for the MuJoCo solvers (gap=0.0), and gap=None keeps Newton's
0.1 m, another mistake of conversion, which the comparison reveals.

The loads of the wheels come from Newton's contact sensor where the solver
reports contact forces (MuJoCo Warp, XPBD), from the MuJoCo data of the
solver on the MuJoCo C path, where Newton 1.7 does not report them, and are
not available with Featherstone and the semi-implicit solver; the drive loop
then shares its force equally between the wheels.
"""
import math

import numpy as np
import warp as wp

import newton

from .base import Kin, PortedCart
from .design import mjcf

SOLVERS = ("mujoco-warp", "mujoco-c", "xpbd", "featherstone", "semi-implicit")
BODIES = ("chassis", "front", "rear", "tool")


class NewtonCart(PortedCart):
    """The cart in Newton; see the module docstring. contact_ke and
    contact_kd, if given, replace the penalty stiffness and damping of every
    shape, which the MJCF text does not carry (Featherstone and the
    semi-implicit solver only)."""
    BACKEND = "newton"

    def __init__(self, design, solver="mujoco-warp", device="cpu", timestep=None,
                 coordinate="axle", contact_ke=None, contact_kd=None, hull_vertices=1000,
                 gap=0.0):
        if solver not in SOLVERS:
            raise ValueError(f"unknown solver {solver!r}; one of {SOLVERS}")
        self.solver_name, self.device = solver, device
        self.hull_vertices, self.gap = hull_vertices, gap
        self.contact_ke, self.contact_kd = contact_ke, contact_kd
        super().__init__(design, timestep, coordinate)

    # --- the model and the solver ---------------------------------------------------
    def _build(self):
        wp.config.quiet = True
        self.xml = mjcf(self.design)
        builder = newton.ModelBuilder()
        builder.add_mjcf(self.xml, mesh_maxhullvert=self.hull_vertices)
        m = self.model = builder.finalize(device=self.device)
        if self.contact_ke is not None:
            m.shape_material_ke.fill_(float(self.contact_ke))
        if self.contact_kd is not None:
            m.shape_material_kd.fill_(float(self.contact_kd))
        if self.solver_name in ("mujoco-warp", "mujoco-c") and self.gap is not None:
            m.shape_gap.fill_(float(self.gap))
        labels = list(m.body_label)
        self.body = {n: next(i for i, s in enumerate(labels) if s.split("/")[-1] == n)
                     for n in BODIES}
        shapes = list(m.shape_label)
        self.ground_shape = next(i for i, s in enumerate(shapes) if s.endswith("/ground"))
        self.wheel_shapes = [next(i for i, s in enumerate(shapes)
                                  if s.endswith(f"/{w}/{w}") and not self._site(i))
                             for w in ("front", "rear")]
        # the order of the joint coordinates and rates: x, z, pitch, front, rear, reach
        if m.joint_coord_count != 6 or m.joint_dof_count != 6:
            raise RuntimeError(f"expected 6 coordinates and 6 rates, got {m.joint_coord_count} "
                               f"and {m.joint_dof_count}")
        self.state_0, self.state_1 = m.state(), m.state()
        self.control = m.control()
        self.contacts, self.pipeline, self.sensor, self.observables = None, None, None, None
        name = self.solver_name
        if name in ("mujoco-warp", "mujoco-c"):
            self.solver = newton.solvers.SolverMuJoCo(m, use_mujoco_cpu=(name == "mujoco-c"))
            if name == "mujoco-warp":
                self._sensor(rigid_contact_max=self.solver.get_max_contact_count())
        else:
            cls = {"xpbd": newton.solvers.SolverXPBD,
                   "featherstone": newton.solvers.SolverFeatherstone,
                   "semi-implicit": newton.solvers.SolverSemiImplicit}[name]
            self.solver = cls(m)
            self.pipeline = newton.CollisionPipeline(m)
            self.contacts = self.pipeline.contacts()
            if name == "xpbd":
                self._sensor()
        newton.eval_fk(m, m.joint_q, m.joint_qd, self.state_0)

    def _site(self, shape):
        return bool(int(self.model.shape_flags.numpy()[shape]) & int(newton.ShapeFlags.SITE))

    def _sensor(self, rigid_contact_max=None):
        from newton.sensors import SensorContact
        m = self.model
        self.sensor = SensorContact(m, sensing_shapes=self.wheel_shapes,
                                    counterpart_shapes=[self.ground_shape],
                                    request_contact_attributes=False)
        if self.pipeline is None:
            self.pipeline = newton.CollisionPipeline(m, rigid_contact_max=rigid_contact_max,
                                                     soft_contact_max=0)
            self.contacts = self.pipeline.contacts()
        self.observables = self.solver.observables(kinds=self.sensor.solver_observable_kinds,
                                                   contacts=self.contacts)

    # --- the operations of a port ------------------------------------------------------
    def _kinematics(self):
        s = self.state_0
        q, qd = s.joint_q.numpy(), s.joint_qd.numpy()
        bq, bqd = s.body_q.numpy(), s.body_qd.numpy()
        f, r = bq[self.body["front"]], bq[self.body["rear"]]
        fv = bqd[self.body["front"]]          # linear velocity of the centre of mass first
        return Kin(front=(float(f[0]), float(f[2])), front_vel=(float(fv[0]), float(fv[2])),
                   rear=(float(r[0]), float(r[2])), pitch=float(q[2]), reach=float(q[5]),
                   front_rate=float(qd[3]), rear_rate=float(qd[4]))

    def _contact_loads(self):
        if self.sensor is not None:
            self.sensor.update(self.state_0, self.contacts, observables=self.observables)
            forces = self.sensor.force_matrix.numpy()[:, 0, :]
            return tuple(self._normal(w, forces[i]) for i, w in enumerate(("front", "rear")))
        if self.solver_name == "mujoco-c":
            return self._mujoco_c_loads()
        return None

    def _normal(self, wheel, force):
        """The component of a contact force along the ground's normal at the
        wheel's foot."""
        nx, nz = self.pr.normal(self.foot(wheel).s)
        return abs(float(force[0] * nx + force[2] * nz))

    def _mujoco_c_loads(self):
        import mujoco
        mm, md = self.solver.mj_model, self.solver.mj_data
        shape_of = self.solver.mjc_geom_to_newton_shape.numpy()[0]
        geom_of = {int(s): g for g, s in enumerate(shape_of)}
        wheels = {geom_of[s]: w for s, w in zip(self.wheel_shapes, ("front", "rear"))}
        ground = geom_of[self.ground_shape]
        total = {"front": 0.0, "rear": 0.0}
        f6 = np.zeros(6)
        for i in range(md.ncon):
            c = md.contact[i]
            pair = {int(c.geom1), int(c.geom2)}
            if ground in pair:
                for g, w in wheels.items():
                    if g in pair:
                        mujoco.mj_contactForce(mm, md, i, f6)
                        total[w] += float(f6[0])
        return total["front"], total["rear"]

    def _write(self, q, qd):
        for s in (self.state_0, self.state_1):
            s.joint_q.assign(np.asarray(q, dtype=np.float32))
            s.joint_qd.assign(np.asarray(qd, dtype=np.float32))
            newton.eval_fk(self.model, s.joint_q, s.joint_qd, s)
        if self.solver_name in ("mujoco-warp", "mujoco-c"):
            # clear MuJoCo's warm start, controls and applied forces, as
            # mj_resetData does in Section 11.1; the joint state is kept
            self.solver.reset(self.state_0, flags=newton.StateFlags.NONE)

    def _apply(self, torque_front, torque_rear, ell):
        f = np.zeros(6, dtype=np.float32)
        f[3], f[4] = torque_front, torque_rear
        self.control.joint_f.assign(f)
        target = np.zeros(6, dtype=np.float32)
        target[5] = ell
        self.control.joint_target_q.assign(target)

    def _step(self):
        s0, s1 = self.state_0, self.state_1
        s0.clear_forces()
        if self.pipeline is not None and self.solver_name not in ("mujoco-warp", "mujoco-c"):
            self.pipeline.collide(s0, self.contacts)
        self.solver.step(s0, s1, self.control, self.contacts, self.dt, observables=self.observables)
        if self.solver_name in ("xpbd", "semi-implicit"):
            # these solvers move the bodies; the joint coordinates follow
            newton.eval_ik(self.model, s1, s1.joint_q, s1.joint_qd)
        self.state_0, self.state_1 = s1, s0

    # --- the record -----------------------------------------------------------------------
    def _hull_size(self):
        src = self.model.shape_source[self.ground_shape]
        verts = getattr(src, "vertices", None)
        return "not reported" if verts is None else str(len(np.asarray(verts)))

    def record(self):
        """What Newton reports of the model and the solver."""
        m = self.model
        out = {
            "bodies, joints, coordinates, rates": f"{m.body_count}, {m.joint_count}, "
                                                 f"{m.joint_coord_count}, {m.joint_dof_count}",
            "joint types": ", ".join(str(newton.JointType(int(t)).name)
                                     for t in m.joint_type.numpy()),
            "body masses [kg]": ", ".join(f"{x:.6g}" for x in m.body_mass.numpy()),
            "wheel inertia about the axle [kg m^2]":
                f"{float(m.body_inertia.numpy()[self.body['front']][1][1]):.6g}",
            "friction of the shapes in contact": ", ".join(
                f"{float(m.shape_material_mu.numpy()[i]):g}"
                for i in (self.ground_shape, *self.wheel_shapes)),
            "tool's joint target gains ke, kd": f"{float(m.joint_target_ke.numpy()[5]):g}, "
                                                f"{float(m.joint_target_kd.numpy()[5]):g}",
            "state precision": str(self.state_0.joint_q.dtype.__name__),
            "vertices of the ground's convex hull": self._hull_size(),
            "gap of the shapes [m]": f"{float(m.shape_gap.numpy()[self.ground_shape]):g}",
        }
        mm = getattr(self.solver, "mj_model", None)
        if mm is not None:
            import mujoco
            o = mm.opt
            names = {int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST): "implicitfast",
                     int(mujoco.mjtIntegrator.mjINT_EULER): "Euler"}
            out["MuJoCo model built by Newton: integrator, cone, solver, iterations, "
                "tolerance"] = (
                f"{names.get(int(o.integrator), int(o.integrator))}, "
                f"{'elliptic' if int(o.cone) == 1 else 'pyramidal'}, "
                f"{'Newton' if int(o.solver) == 2 else int(o.solver)}, {int(o.iterations)}, "
                f"{float(o.tolerance):.9g}")
            flags = [n[len("mjDSBL_"):].lower() for n in dir(mujoco.mjtDisableBit)
                     if n.startswith("mjDSBL_")
                     and int(o.disableflags) & int(getattr(mujoco.mjtDisableBit, n))]
            out["MuJoCo model built by Newton: disabled features"] = ", ".join(flags) or "none"
            out["MuJoCo model built by Newton: ground mesh vertices, faces"] = (
                f"{int(mm.mesh_vertnum[0])}, {int(mm.mesh_facenum[0])}")
            out["MuJoCo model built by Newton: gravity [m/s^2]"] = f"{float(o.gravity[2]):.9g}"
        return out
