"""The MuJoCo model of the cart on its track, generated from a design
(Section 11.1.1). Every number of the model is a field of Design, and the
model is written as MJCF text by mjcf(), so that the text that MuJoCo
compiles can be read, hashed and archived."""
import math
from dataclasses import asdict, dataclass

from .track import Profile

VARIANTS = ("wall in radians", "speed servos", "welded wheels")


@dataclass(frozen=True)
class Design:
    """All parameters of the simulated cart and track, in SI units."""
    # track
    ramp_start: float = 20.0        # ground coordinate at which the crest begins [m]
    wall: float = 25.0              # track coordinate p + l at which the tool touches the wall [m]
    slope_deg: float = 9.5          # angle of the slope [deg]
    crest_radius: float = 10.0      # radius of the crest [m]
    crest_facets: int = 166         # flat facets of the crest in the ground's mesh
    friction: float = 0.25          # sliding friction between wheels and ground
    # cart
    chassis_mass: float = 80.0      # [kg]
    wheel_mass: float = 5.0         # each of two wheels [kg]
    tool_mass: float = 0.1          # [kg]
    half_length: float = 0.35       # half length of the chassis [m]
    wheel_offset: float = 0.25      # axles 0.25 m ahead of and behind the centre [m]
    wheel_drop: float = 0.05        # axles 0.05 m below the centre [m]
    wheel_radius: float = 0.1       # [m]
    reach: float = 0.8              # reach of the unfolded tool, and stroke of its slide [m]
    # actuators
    drive_limit: float = 40.0       # torque limit of each wheel's motor [N m]
    speed_gain: float = 100.0       # gain of the drive loop on the cart's speed [1/s]
    servo_gain: float = 45.5        # the variant "speed servos" only [N m s/rad]
    tool_kp: float = 1000.0         # tool position servo [N/m]
    tool_kv: float = 20.0           # tool position servo [N s/m]
    tool_force: float = 100.0       # tool force limit [N]
    # simulator
    timestep: float = 0.001         # [s]
    integrator: str = "implicitfast"
    solver: str = "Newton"
    cone: str = "elliptic"
    iterations: int = 100
    tolerance: float = 1e-8

    def profile(self):
        return Profile(self.ramp_start, self.crest_radius, math.radians(self.slope_deg),
                       self.wheel_radius)

    def as_dict(self):
        return asdict(self)


def num(x):
    """A number for the MJCF text: at most 12 significant digits."""
    return f"{x:.12g}"


def ground_profile(d):
    """The points (x, z) of the ground's mesh along the profile: 5 m of firm
    ground behind the start, the crest as d.crest_facets chords whose ends
    lie on the circle, and the slope to 8 m beyond the crest."""
    pr = d.profile()
    pts = [(-5.0, 0.0), (d.ramp_start, 0.0)]
    for i in range(1, d.crest_facets + 1):
        x, z, _ = pr.point(pr.s0 + (pr.s1 - pr.s0) * i / d.crest_facets)
        pts.append((x, z))
    x, z, _ = pr.point(pr.s1 + 8.0)
    pts.append((x, z))
    return pts


def mjcf(d=Design(), variant=None):
    """The model as MJCF text. The ground is one convex mesh, the prism
    under the profile, so that no two pieces of ground overlap where the
    profile changes. The compiler reads angles in degrees, its default,
    which the file states; the wall is rotated by a quaternion, which carries
    no unit. The variants are those of the diagnostics of Section 11.1.3:
    "wall in radians" writes the wall's rotation as an Euler angle in
    radians, a mistake; "speed servos" adds to each wheel MuJoCo's velocity
    actuator as a stiff speed servo, another; "welded wheels" welds the
    wheels to the chassis by equality constraints, for a cart that slides."""
    if variant is not None and variant not in VARIANTS:
        raise ValueError(variant)
    pr = d.profile()
    th = pr.theta
    c, s = math.cos(th), math.sin(th)
    if variant == "wall in radians":
        rot = f'euler="0 {num(th)} 0"'
    else:
        rot = f'quat="{num(math.cos(th / 2))} 0 {num(math.sin(th / 2))} 0"'
    pts = ground_profile(d)
    x_end, z_end = pts[-1]
    verts = []
    for x, z in pts + [(-5.0, z_end - 1.0), (x_end, z_end - 1.0)]:
        verts += [x, -1.0, z, x, 1.0, z]
    vertex = " ".join(num(v) for v in verts)
    # the wall's face, perpendicular to the slope, where the tool tip touches
    # it at p + l = wall: beyond the crest the track coordinate counts the
    # path of the axle, longer than the ground by r * theta
    wx, wz, _ = pr.point(d.wall - d.wheel_radius * th)
    T, W = 0.05, 0.5                                     # wall: half thickness, half height
    wcx, wcz = wx + T * c + W * s, wz - T * s + W * c
    hl, wo, wd, r = d.half_length, d.wheel_offset, d.wheel_drop, d.wheel_radius
    lim = d.drive_limit
    servos = equality = ""
    if variant == "speed servos":
        servos = "".join(f"""
    <velocity name="speed {w}" joint="{w}" kv="{num(d.servo_gain)}"
              forcerange="{num(-lim)} {num(lim)}"/>""" for w in ("front", "rear"))
    if variant == "welded wheels":
        equality = """
  <equality>
    <joint joint1="front"/>
    <joint joint1="rear"/>
  </equality>"""
    return f"""<mujoco model="cart on track">
  <compiler angle="degree"/>
  <option timestep="{num(d.timestep)}" integrator="{d.integrator}" solver="{d.solver}"
          cone="{d.cone}" iterations="{d.iterations}" tolerance="{num(d.tolerance)}"/>
  <default>
    <geom friction="{num(d.friction)} 0.005 0.0001"/>
  </default>
  <asset>
    <mesh name="ground" vertex="{vertex}"/>
  </asset>
  <worldbody>
    <geom name="ground" type="mesh" mesh="ground"/>
    <geom name="wall" type="box" pos="{num(wcx)} 0 {num(wcz)}" size="{num(T)} 1 {num(W)}"
          {rot}
          contype="2" conaffinity="2"/>
    <body name="chassis" pos="{num(-hl)} 0 {num(r + wd)}">
      <joint name="x" type="slide" axis="1 0 0"/>
      <joint name="z" type="slide" axis="0 0 1"/>
      <joint name="pitch" type="hinge" axis="0 1 0"/>
      <geom name="chassis" type="box" size="{num(hl)} 0.25 0.08" mass="{num(d.chassis_mass)}"
            contype="2" conaffinity="2"/>
      <site name="imu"/>
      <site name="bumper" pos="{num(hl)} 0 0"/>
      <body name="front" pos="{num(wo)} 0 {num(-wd)}">
        <joint name="front" type="hinge" axis="0 1 0"/>
        <geom name="front" type="sphere" size="{num(r)}" mass="{num(d.wheel_mass)}"/>
        <site name="front" type="sphere" size="{num(r + 0.005)}"/>
      </body>
      <body name="rear" pos="{num(-wo)} 0 {num(-wd)}">
        <joint name="rear" type="hinge" axis="0 1 0"/>
        <geom name="rear" type="sphere" size="{num(r)}" mass="{num(d.wheel_mass)}"/>
        <site name="rear" type="sphere" size="{num(r + 0.005)}"/>
      </body>
      <body name="tool" pos="{num(hl)} 0 0">
        <joint name="reach" type="slide" axis="1 0 0" range="0 {num(d.reach)}"/>
        <geom name="tool" type="box" pos="-0.05 0 0" size="0.05 0.02 0.02"
              mass="{num(d.tool_mass)}" contype="2" conaffinity="2"/>
        <site name="tip" euler="0 90 0"/>
      </body>
    </body>
  </worldbody>{equality}
  <actuator>
    <motor name="drive front" joint="front" ctrlrange="{num(-lim)} {num(lim)}"/>
    <motor name="drive rear" joint="rear" ctrlrange="{num(-lim)} {num(lim)}"/>{servos}
    <position name="tool" joint="reach" kp="{num(d.tool_kp)}" kv="{num(d.tool_kv)}"
              ctrlrange="0 {num(d.reach)}" forcerange="{num(-d.tool_force)} {num(d.tool_force)}"/>
  </actuator>
  <sensor>
    <jointpos name="front angle" joint="front"/>
    <jointpos name="rear angle" joint="rear"/>
    <jointvel name="front rate" joint="front"/>
    <jointvel name="rear rate" joint="rear"/>
    <jointpos name="reach" joint="reach"/>
    <framepos name="bumper position" objtype="site" objname="bumper"/>
    <framelinvel name="bumper velocity" objtype="site" objname="bumper"/>
    <accelerometer name="accelerometer" site="imu"/>
    <rangefinder name="range" site="tip"/>
    <touch name="front load" site="front"/>
    <touch name="rear load" site="rear"/>
    <actuatorfrc name="drive front torque" actuator="drive front"/>
    <actuatorfrc name="drive rear torque" actuator="drive rear"/>
    <actuatorfrc name="tool force" actuator="tool"/>
  </sensor>
</mujoco>
"""
