"""The simulated cart in Isaac Sim (Section 11.2): the stage, the track
coordinate, placement at a state, and the drive loop that realizes a
commanded acceleration over one period of the governor.

The drive loop is the one of Section 11.1, line for line; only the access
to the simulator differs. The cart is a PhysX articulation, read and written
through the tensor API with NumPy on the CPU: link poses and velocities,
joint (DOF) positions, velocities, efforts and drive targets, and the
contact forces between each wheel and the ground. Physics is stepped one
time step at a time by the simulation manager, never by an update of the
application, which would step it by its own clock. Angles in the tensor
API are in radians; on the stage, in degrees.

The few calls that differ between Isaac Sim 5.1 and 6.x are collected in
Physics."""
import inspect
import math

import numpy as np

import carb
import omni.timeline
import omni.usd

from . import stage as stage_mod
from .design import Design, IsaacSettings

G = 9.81               # gravity of the stage [m/s^2]
V_REST = 5e-3          # below 5 mm/s the drive loop counts the cart as at rest [m/s]
APP = None             # the SimulationApp; the script sets it before any IsaacCart is made


class Physics:
    """Stepping, the simulation device and the tensor views, for Isaac Sim
    5.1 (Python 3.11) and 6.x (Python 3.12)."""

    def __init__(self):
        from isaacsim.core.simulation_manager import SimulationManager
        self.sm = SimulationManager
        self.step_takes_count = "steps" in inspect.signature(SimulationManager.step).parameters
        self.own_view = False
        self.mode, self.scene_dt, self.dt, self._t = None, None, None, 0.0

    def configure(self, dt):
        """Steps through the simulation manager if it steps by dt, and
        directly through the physics interface with dt otherwise."""
        self.dt = dt
        try:
            self.scene_dt = float(self.sm.get_physics_dt())
        except Exception:                          # noqa: BLE001
            self.scene_dt = None
        if self.scene_dt is not None and abs(self.scene_dt - dt) < 1e-12:
            self.mode = "simulation manager"
            return
        self.mode = "physics interface"
        try:
            import omni.physics.core as core       # 6.x
            self._direct, self._fetch = core.get_physics_simulation_interface(), False
        except ImportError:
            from omni.physx import get_physx_simulation_interface   # 5.1
            self._direct, self._fetch = get_physx_simulation_interface(), True

    def use_cpu(self):
        if hasattr(self.sm, "set_device"):
            self.sm.set_device("cpu")              # 6.x
        else:
            self.sm.set_physics_sim_device("cpu")  # 5.1

    def step(self):
        """One physics step of dt."""
        if self.mode == "simulation manager":
            if self.step_takes_count:
                self.sm.step(steps=1)              # 6.x
            else:
                self.sm.step()                     # 5.1: simulate and fetch the results
            return
        self._direct.simulate(self.dt, self._t)
        if self._fetch:
            self._direct.fetch_results()
        self._t += self.dt

    def view(self):
        """A tensor simulation view with the NumPy frontend."""
        view = None
        try:
            view = self.sm.get_physics_sim_view()
        except Exception:                          # noqa: BLE001 (absent in a later version)
            view = None
        if view is None or not getattr(view, "is_valid", True):
            import omni.physics.tensors as tensors
            stage_id = omni.usd.get_context().get_stage_id()
            try:
                view = tensors.create_simulation_view("numpy", stage_id=stage_id, backend="physx")
            except TypeError:
                view = tensors.create_simulation_view("numpy", stage_id=stage_id)
            view.set_subspace_roots("/")
            self.own_view = True
        return view

    def time(self):
        return float(self.sm.get_simulation_time())

    def count(self):
        return int(self.sm.get_num_physics_steps())


def scene_query():
    from omni.physx import get_physx_scene_query_interface
    return get_physx_scene_query_interface()


class IsaacCart:
    """The cart on its track in Isaac Sim. variant is one of
    stage.VARIANTS; coordinate="ground" is the mistake of Section S11.9.1,
    following the ground coordinate."""

    def __init__(self, design=Design(), settings=IsaacSettings(), variant=None, coordinate="axle"):
        self.design, self.settings, self.variant = design, settings, variant
        self.coordinate = coordinate
        self.pr = design.profile()
        self.physics = Physics()
        self.servos = variant in ("speed servos", "servo damping per radian")
        self.lead = design.half_length - design.wheel_offset    # bumper ahead of the axle
        self.dt = settings.timestep
        self.steps = int(round(0.1 / self.dt))                  # steps per period
        self.wheel_inertia = 0.4 * design.wheel_mass * design.wheel_radius ** 2
        # the mass that the wheels drive along the track: the cart and both auxiliary links
        self.mass = (design.chassis_mass + 2 * design.wheel_mass + design.tool_mass
                     + 2 * settings.aux_mass)
        self.clock = 0                                          # physics steps taken
        self._cache = None
        self._open()
        self.sink = 0.0
        self.sink = self.static_sink()

    # --- the stage and the views ------------------------------------------------------
    def _open(self):
        ctx = omni.usd.get_context()
        timeline = omni.timeline.get_timeline_interface()
        if not timeline.is_stopped():
            timeline.stop()
            APP.update()
        ctx.new_stage()
        APP.update()
        self.stage = ctx.get_stage()
        self.paths = stage_mod.build(self.stage, self.design, self.settings, self.variant)
        APP.update()
        self.physics.use_cpu()
        timeline.play()
        APP.update()
        self.physics.configure(self.dt)
        self.view = self.physics.view()
        self.art = self.view.create_articulation_view(self.paths["root"])
        if self.art is None or not self.art.check():
            raise RuntimeError(f"no valid articulation view at {self.paths['root']}")
        meta = self.art.shared_metatype
        self.dof_names, self.link_names = list(meta.dof_names), list(meta.link_names)
        self.dof = {n: self.dof_names.index(n) for n in stage_mod.JOINTS}
        self.ndof = len(self.dof_names)
        self.idx = np.array([0], dtype=np.int32)
        try:                                       # link poses from the articulation view
            self.art.get_link_transforms()
            self.art.get_link_velocities()
            self.links_from, self.bodies = "articulation view", None
            self.link = {n: self.link_names.index(n) for n in stage_mod.LINKS}
        except Exception:                          # noqa: BLE001 (or from a rigid-body view)
            self.bodies = self.view.create_rigid_body_view([self.paths[n] for n in stage_mod.LINKS])
            self.links_from = "rigid body view"
            self.link = {n: i for i, n in enumerate(stage_mod.LINKS)}
        self.contacts = {w: self._contact_view(w) for w in ("front", "rear")}

    def _contact_view(self, wheel):
        path, ground = self.paths[wheel], self.paths["ground"]
        error = None
        for pattern, filters in (([path], [[ground]]), (path, [ground]), ([path], [ground])):
            try:
                view = self.view.create_rigid_contact_view(pattern, filter_patterns=filters,
                                                           max_contact_data_count=64)
                if view is not None and view.check():
                    return view
            except Exception as exc:               # noqa: BLE001 (try the next form)
                error = exc
        raise RuntimeError(f"no rigid contact view for {path}: {error!r}")

    def close(self):
        timeline = omni.timeline.get_timeline_interface()
        self.contacts, self.art = {}, None
        if self.physics.own_view and self.view is not None:
            try:
                self.view.invalidate()
            except Exception:                      # noqa: BLE001
                pass
        self.view = None
        timeline.stop()
        APP.update()

    def _kinematics(self):
        """After the joint state is written, update the link poses."""
        self._cache = None
        try:
            self.view.update_articulations_kinematic()
        except Exception:                          # noqa: BLE001 (not in every version)
            pass

    # --- readings -----------------------------------------------------------------------
    def _read(self):
        if self._cache is None:
            if self.bodies is None:
                links = np.asarray(self.art.get_link_transforms(), dtype=float)[0]
                vels = np.asarray(self.art.get_link_velocities(), dtype=float)[0]
            else:
                links = np.asarray(self.bodies.get_transforms(), dtype=float)
                vels = np.asarray(self.bodies.get_velocities(), dtype=float)
            self._cache = (links, vels,
                           np.asarray(self.art.get_dof_positions(), dtype=float)[0],
                           np.asarray(self.art.get_dof_velocities(), dtype=float)[0])
        return self._cache

    def position(self, link):
        """World position (x, y, z) of a link."""
        return tuple(self._read()[0][self.link[link], :3])

    def axis_x(self, link="chassis"):
        """The x axis of a link in the world frame."""
        qx, qy, qz, qw = self._read()[0][self.link[link], 3:7]
        return (1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy + qw * qz), 2 * (qx * qz - qw * qy))

    def pitch(self):
        ax = self.axis_x("chassis")
        return math.atan2(-ax[2], ax[0])

    def foot(self, wheel="front"):
        x, _, z = self.position(wheel)
        return self.pr.foot(x, z)

    def state(self, coordinate=None):
        """(p, v, l): the track coordinate of the bumper, its rate, and the
        reach of the tool."""
        links, vels, dofs, _ = self._read()
        i = self.link["front"]
        f = self.pr.foot(links[i, 0], links[i, 2])
        vx, vz = vels[i, 0], vels[i, 2]
        if (coordinate or self.coordinate) == "ground":
            p, v = f.s + self.lead, self.pr.ground_rate(f, vx, vz)
        else:
            p, v = self.pr.axle(f.s) + self.lead, self.pr.axle_rate(f, vx, vz)
        return float(p), float(v), float(dofs[self.dof["reach"]])

    def axle_speed(self):
        """The speed of the front axle along the ground's tangent at its
        foot. On the crest, the rate of p exceeds it by the factor
        (R + r) / (R + d), where d, the axle's height above the ground, is
        r less the penetration."""
        links, vels, _, _ = self._read()
        i = self.link["front"]
        f = self.pr.foot(links[i, 0], links[i, 2])
        return float(vels[i, 0] * math.cos(f.angle) - vels[i, 2] * math.sin(f.angle))

    def wheel_rates(self):
        dofvel = self._read()[3]
        return float(dofvel[self.dof["front"]]), float(dofvel[self.dof["rear"]])

    def contact_forces(self):
        """The contact force of the ground on each wheel in the last step,
        in the world frame [N]."""
        out = {}
        for wheel, view in self.contacts.items():
            f = np.asarray(view.get_contact_force_matrix(self.dt), dtype=float).reshape(-1)
            out[wheel] = f[:3] if f.size >= 3 else np.zeros(3)
        return out

    def loads(self):
        """The load on each wheel: its contact force along the ground's
        normal at the wheel's foot."""
        forces = self.contact_forces()
        out = []
        for wheel in ("front", "rear"):
            nx, nz = self.pr.normal(self.foot(wheel).s)
            f = forces[wheel]
            out.append(abs(float(f[0] * nx + f[2] * nz)))
        return out

    def separations(self):
        """The least separation of each wheel's contacts with the ground in
        the last step (negative: penetration), or None without contact."""
        out = {}
        for wheel, view in self.contacts.items():
            data = view.get_contact_data(self.dt)
            dist, counts = np.asarray(data[3], dtype=float).reshape(-1), np.asarray(data[4]).reshape(-1)
            starts = np.asarray(data[5]).reshape(-1)
            n, s0 = int(counts[0]), int(starts[0])
            out[wheel] = float(dist[s0:s0 + n].min()) if n > 0 else None
        return out

    def raycast(self, origin, direction, distance=10.0):
        """(distance, collider path) of the closest hit, or (None, None)."""
        hit = scene_query().raycast_closest(carb.Float3(*[float(a) for a in origin]),
                                            carb.Float3(*[float(a) for a in direction]),
                                            float(distance))
        if not hit or not hit.get("hit"):
            return None, None
        return float(hit["distance"]), str(hit.get("collision", ""))

    # --- placement ------------------------------------------------------------------------
    def _axle(self, s):
        x, z, _ = self.pr.point(s)
        nx, nz = self.pr.normal(s)
        h = self.design.wheel_radius - self.sink
        return x + h * nx, z + h * nz

    def _pose(self, p):
        """Chassis position (x, z) and pitch with the bumper at track
        coordinate p and both wheels on the ground (as in Section 11.1)."""
        sf = self.pr.ground(p - self.lead)
        F = self._axle(sf)
        base = 2 * self.design.wheel_offset
        lo, hi = sf - base - 0.2, sf
        for _ in range(80):
            mid = (lo + hi) / 2
            R = self._axle(mid)
            if math.hypot(F[0] - R[0], F[1] - R[1]) > base:
                lo = mid
            else:
                hi = mid
        R = self._axle((lo + hi) / 2)
        pitch = math.atan2(-(F[1] - R[1]), F[0] - R[0])
        wd = self.design.wheel_drop
        return ((F[0] + R[0]) / 2 + wd * math.sin(pitch),
                (F[1] + R[1]) / 2 + wd * math.cos(pitch), pitch)

    def place(self, p, v, l):
        """Put the cart at the state (p, v, l), rolling without slip at
        speed v, its wheels sunk by the static penetration, the tool still."""
        d = self.design
        cx, cz, pitch = self._pose(p)
        eps = 1e-6
        a, b = self._pose(p + eps), self._pose(p - eps)
        q = np.zeros((1, self.ndof), dtype=np.float32)
        qd = np.zeros((1, self.ndof), dtype=np.float32)
        q[0, self.dof["x"]] = cx + d.half_length
        q[0, self.dof["z"]] = cz - (d.wheel_radius + d.wheel_drop)
        q[0, self.dof["pitch"]] = pitch
        q[0, self.dof["reach"]] = l
        rates = [v * (a[i] - b[i]) / (2 * eps) for i in range(3)]
        for name, rate in zip(("x", "z", "pitch"), rates):
            qd[0, self.dof[name]] = rate
        if self.variant != "welded wheels":
            qd[0, self.dof["front"]] = qd[0, self.dof["rear"]] = v / d.wheel_radius - rates[2]
        self.art.set_dof_positions(q, self.idx)
        self.art.set_dof_velocities(qd, self.idx)
        target = np.zeros((1, self.ndof), dtype=np.float32)
        target[0, self.dof["reach"]] = l
        self.art.set_dof_position_targets(target, self.idx)
        self.art.set_dof_velocity_targets(np.zeros((1, self.ndof), dtype=np.float32), self.idx)
        self.art.set_dof_actuation_forces(np.zeros((1, self.ndof), dtype=np.float32), self.idx)
        self._kinematics()

    def static_sink(self):
        """Penetration of the wheels at rest on the firm ground, from the
        height of the front axle (zero would mean that PhysX keeps the
        wheels exactly on the surface)."""
        self.place(5.0, 0.0, 0.0)
        for _ in range(30):
            self.period(0.0, 0.0)
        return self.design.wheel_radius - self.position("front")[2]

    # --- the drive loop -------------------------------------------------------------------
    def step(self):
        self.physics.step()
        self.clock += 1
        self._cache = None

    def controls(self, a_ref, v_ref, ell):
        """The drive loop of Section 11.1 for the current state: a force
        M (a_ref - g sin(pitch)) and a feedback force M K (v_ref - v),
        shared between the wheels' motors in proportion to their loads, and
        each wheel what its own inertia needs; the tool's drive is set to
        the reach ell. In the servo variants, the wheels' velocity drives
        provide the feedback instead."""
        d, r = self.design, self.design.wheel_radius
        pitch = self.pitch()
        v = self.state()[1]
        gain = 0.0 if self.servos else d.speed_gain
        force = self.mass * (a_ref - G * math.sin(pitch) + gain * (v_ref - v))
        loads = self.loads()
        total, lim = sum(loads), d.drive_limit
        effort = np.zeros((1, self.ndof), dtype=np.float32)
        for wheel, load in zip(("front", "rear"), loads):
            share = load / total if total > 0 else 0.5
            tau = force * r * share + self.wheel_inertia * a_ref / r
            effort[0, self.dof[wheel]] = max(-lim, min(lim, tau))
        self.art.set_dof_actuation_forces(effort, self.idx)
        if self.servos:
            rates = self.wheel_rates()
            target = np.zeros((1, self.ndof), dtype=np.float32)
            for wheel, rate in zip(("front", "rear"), rates):
                target[0, self.dof[wheel]] = rate + a_ref * self.dt / r + (v_ref - v) / r
            self.art.set_dof_velocity_targets(target, self.idx)
        target = np.zeros((1, self.ndof), dtype=np.float32)
        target[0, self.dof["reach"]] = ell
        self.art.set_dof_position_targets(target, self.idx)

    def period(self, u, ell, kind="segment", record=None,
               steps=None):
        """One period of 0.1 s under the commanded acceleration u
        and the commanded reach ell, held over the period; or the
        first 'steps' steps of one."""
        v0 = self.state()[1]
        if abs(v0) < V_REST:
            v0 = 0.0                     # at rest: start from zero
        if kind == "rest":               # brake, then hold
            brake = 1 if v0 > 0 else 0
        else:
            brake = 1 if v0 > 0 > u else -1 if v0 < 0 < u else 0
        for i in range(steps or self.steps):
            tau = i * self.dt
            v_ref, a_ref = v0 + u * tau, u
            if (kind == "rest" and v0 <= 0) or (
                    brake and v_ref * v0 <= 0):
                v_ref, a_ref = 0.0, 0.0  # at rest: hold
            if record is not None:
                record.append(self.sample(a_ref, v_ref))
            self.controls(a_ref, v_ref, ell)
            self.step()                  # one physics step
        return self.state()

    def sample(self, a_ref, v_ref):
        """The signals at the current step, for the records."""
        p, v, l = self.state()
        front, rear = self.wheel_rates()
        loads = self.loads()
        return dict(t=self.clock * self.dt, p=p, v=v, l=l, a_ref=a_ref, v_ref=v_ref,
                    axle=self.axle_speed(), front_rate=front, rear_rate=rear, pitch=self.pitch(),
                    front_load=loads[0], rear_load=loads[1])
