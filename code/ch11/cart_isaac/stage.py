"""The cart and its track as a USD stage for Isaac Sim (Section 11.2),
authored from the same design as the MuJoCo model of Section 11.1.

Every number comes from Design or IsaacSettings, and the stage is written
with the USD and PhysX schemas only, so that the exported text can be read,
hashed and archived like the MJCF text of Section 11.1. Units follow USD:
lengths in stage units (metres here, metersPerUnit = 1), masses in
kilograms, and angles in degrees, both for rotations of prims and for the
limits and drive gains of revolute joints.

The chassis moves in the vertical plane of the track. MuJoCo gives one body
three joints; a PhysX articulation needs one link per joint, so the planar
joint is a chain of two auxiliary links, base -> slider_x (slide along x)
-> slider_z (slide along z) -> chassis (hinge about y), whose masses
(IsaacSettings.aux_mass) are reported with the record."""
import json
import math

from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

from .design import Design, IsaacSettings, ground_profile

G = 9.81
VARIANTS = ("wall in radians", "stage in centimetres", "speed servos",
            "servo damping per radian", "welded wheels", "physx defaults")
LINKS = ("base", "slider_x", "slider_z", "chassis", "front", "rear", "tool")
JOINTS = ("x", "z", "pitch", "front", "rear", "reach")


def box_inertia(m, sx, sy, sz):
    """Moments of inertia of a uniform box of full sizes sx, sy, sz, as
    MuJoCo computes them from the geoms of Section 11.1."""
    return (m * (sy * sy + sz * sz) / 12, m * (sx * sx + sz * sz) / 12,
            m * (sx * sx + sy * sy) / 12)


def _xform(prim, translate=None, orient=None, rotate_y=None, scale=None):
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    if translate is not None:
        xf.AddTranslateOp().Set(Gf.Vec3d(*translate))
    if orient is not None:
        xf.AddOrientOp().Set(Gf.Quatf(orient[0], Gf.Vec3f(*orient[1:])))
    if rotate_y is not None:
        xf.AddRotateYOp().Set(float(rotate_y))
    if scale is not None:
        xf.AddScaleOp().Set(Gf.Vec3f(*scale))


def _collider(prim, s, variant):
    UsdPhysics.CollisionAPI.Apply(prim)
    if variant != "physx defaults":
        api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        api.CreateContactOffsetAttr(s.contact_offset)
        api.CreateRestOffsetAttr(s.rest_offset)


def _material(stage, path, mu):
    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(mu)
    api.CreateDynamicFrictionAttr(mu)
    api.CreateRestitutionAttr(0.0)
    return mat


def _bind(prim, mat):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def _link(stage, path, pos, mass, inertia, s, variant, com=(0.0, 0.0, 0.0)):
    link = UsdGeom.Xform.Define(stage, path)
    prim = link.GetPrim()
    _xform(prim, translate=pos)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    m = UsdPhysics.MassAPI.Apply(prim)
    m.CreateMassAttr(float(mass))
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
    m.CreateCenterOfMassAttr(Gf.Vec3f(*com))
    m.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    if variant != "physx defaults":
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateLinearDampingAttr(s.damping)
        rb.CreateAngularDampingAttr(s.damping)
    return prim


def _joint(stage, kind, path, body0, body1, pos0, pos1=(0.0, 0.0, 0.0), axis=None):
    joint = {"fixed": UsdPhysics.FixedJoint, "prismatic": UsdPhysics.PrismaticJoint,
             "revolute": UsdPhysics.RevoluteJoint}[kind].Define(stage, path)
    if body0 is not None:
        joint.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*pos0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    joint.CreateLocalPos1Attr(Gf.Vec3f(*pos1))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    if axis is not None:
        joint.CreateAxisAttr(axis)
    return joint


def build(stage, d=Design(), s=IsaacSettings(), variant=None):
    """Authors the physics scene, the track and the cart on an empty stage,
    with the cart at p = 0, at rest, its tool folded. Returns the paths of
    the prims that the simulator adapter reads."""
    if variant is not None and variant not in VARIANTS:
        raise ValueError(variant)
    pr = d.profile()
    th = pr.theta
    c, sn = math.cos(th), math.sin(th)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01 if variant == "stage in centimetres" else 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    # --- the physics scene ----------------------------------------------------------
    scene = UsdPhysics.Scene.Define(stage, "/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    if variant != "stage in centimetres":   # the variant keeps the default, Earth's gravity in stage units
        scene.CreateGravityMagnitudeAttr(G)
    px = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    px.CreateTimeStepsPerSecondAttr(int(round(1.0 / s.timestep)))
    px.CreateSolverTypeAttr(s.solver)
    px.CreateEnableGPUDynamicsAttr(s.gpu_dynamics)
    px.CreateBroadphaseTypeAttr(s.broadphase)
    px.CreateEnableCCDAttr(False)
    px.CreateEnableStabilizationAttr(False)
    try:
        px.CreateFrictionTypeAttr(s.friction_type)
    except AttributeError:                  # not in every version of the schema
        pass

    # --- materials --------------------------------------------------------------------
    ground_mat = _material(stage, "/World/materials/ground", d.friction)
    wheel_mat = _material(stage, "/World/materials/wheel", d.friction)

    # --- the ground: the surface of the profile, as a triangle mesh -------------------
    pts = ground_profile(d)
    points, counts, indices = [], [], []
    for x, z in pts:
        points += [Gf.Vec3f(x, -1.0, z), Gf.Vec3f(x, 1.0, z)]
    for i in range(len(pts) - 1):
        a, b = 2 * i, 2 * (i + 1)
        counts.append(4)
        indices += [a, b, b + 1, a + 1]     # counter-clockwise seen from above
    ground = UsdGeom.Mesh.Define(stage, "/World/ground")
    ground.CreatePointsAttr(Vt.Vec3fArray(points))
    ground.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    ground.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
    ground.CreateSubdivisionSchemeAttr("none")
    _collider(ground.GetPrim(), s, variant)
    UsdPhysics.MeshCollisionAPI.Apply(ground.GetPrim()).CreateApproximationAttr("none")
    _bind(ground.GetPrim(), ground_mat)

    # --- the wall: its face perpendicular to the slope at p + l = wall -----------------
    wx, wz, _ = pr.point(d.wall - d.wheel_radius * th)
    T, W = 0.05, 0.5                                      # half thickness, half height
    wall = UsdGeom.Cube.Define(stage, "/World/wall")
    wall.CreateSizeAttr(1.0)
    if variant == "wall in radians":
        _xform(wall.GetPrim(), translate=(wx + T * c + W * sn, 0.0, wz - T * sn + W * c),
               rotate_y=th, scale=(2 * T, 2.0, 2 * W))   # radians where USD reads degrees
    else:
        _xform(wall.GetPrim(), translate=(wx + T * c + W * sn, 0.0, wz - T * sn + W * c),
               orient=(math.cos(th / 2), 0.0, math.sin(th / 2), 0.0), scale=(2 * T, 2.0, 2 * W))
    _collider(wall.GetPrim(), s, variant)

    # --- the cart, at p = 0 -------------------------------------------------------------
    hl, wo, wd, r = d.half_length, d.wheel_offset, d.wheel_drop, d.wheel_radius
    c0 = (-hl, 0.0, r + wd)                               # chassis centre
    root = UsdGeom.Xform.Define(stage, "/World/cart").GetPrim()
    root.SetCustomDataByKey("physai:design", json.dumps(d.as_dict()))
    root.SetCustomDataByKey("physai:settings", json.dumps(s.as_dict()))
    root.SetCustomDataByKey("physai:variant", variant or "")
    UsdPhysics.ArticulationRootAPI.Apply(root)
    art = PhysxSchema.PhysxArticulationAPI.Apply(root)
    art.CreateEnabledSelfCollisionsAttr(False)
    art.CreateSolverPositionIterationCountAttr(s.position_iterations)
    art.CreateSolverVelocityIterationCountAttr(s.velocity_iterations)
    if variant != "physx defaults":
        art.CreateSleepThresholdAttr(0.0)
        art.CreateStabilizationThresholdAttr(0.0)
    aux = (s.aux_inertia,) * 3
    for name in ("base", "slider_x", "slider_z"):
        _link(stage, f"/World/cart/{name}", c0, s.aux_mass, aux, s, variant)
    chassis = _link(stage, "/World/cart/chassis", c0, d.chassis_mass,
                    box_inertia(d.chassis_mass, 2 * hl, 0.5, 0.16), s, variant)
    box = UsdGeom.Cube.Define(stage, "/World/cart/chassis/collider")
    box.CreateSizeAttr(1.0)
    _xform(box.GetPrim(), scale=(2 * hl, 0.5, 0.16))
    _collider(box.GetPrim(), s, variant)
    wheel_inertia = (0.4 * d.wheel_mass * r * r,) * 3
    for name, dx in (("front", wo), ("rear", -wo)):
        link = _link(stage, f"/World/cart/{name}", (c0[0] + dx, 0.0, c0[2] - wd), d.wheel_mass,
                     wheel_inertia, s, variant)
        sphere = UsdGeom.Sphere.Define(stage, f"/World/cart/{name}/collider")
        sphere.CreateRadiusAttr(r)
        _collider(sphere.GetPrim(), s, variant)
        _bind(sphere.GetPrim(), wheel_mat)
        PhysxSchema.PhysxContactReportAPI.Apply(link).CreateThresholdAttr(0.0)
    tool = _link(stage, "/World/cart/tool", (c0[0] + hl, 0.0, c0[2]), d.tool_mass,
                 box_inertia(d.tool_mass, 0.1, 0.04, 0.04), s, variant, com=(-0.05, 0.0, 0.0))
    tbox = UsdGeom.Cube.Define(stage, "/World/cart/tool/collider")
    tbox.CreateSizeAttr(1.0)
    _xform(tbox.GetPrim(), translate=(-0.05, 0.0, 0.0), scale=(0.1, 0.04, 0.04))
    _collider(tbox.GetPrim(), s, variant)

    # --- joints: the base fixed to the world, the planar joint, wheels and tool ---------
    J = "/World/cart/joints"
    _joint(stage, "fixed", f"{J}/root", None, "/World/cart/base", c0)
    _joint(stage, "prismatic", f"{J}/x", "/World/cart/base", "/World/cart/slider_x",
           (0.0, 0.0, 0.0), axis="X")
    _joint(stage, "prismatic", f"{J}/z", "/World/cart/slider_x", "/World/cart/slider_z",
           (0.0, 0.0, 0.0), axis="Z")
    _joint(stage, "revolute", f"{J}/pitch", "/World/cart/slider_z", "/World/cart/chassis",
           (0.0, 0.0, 0.0), axis="Y")
    lim = d.drive_limit
    for name, dx in (("front", wo), ("rear", -wo)):
        joint = _joint(stage, "revolute", f"{J}/{name}", "/World/cart/chassis",
                       f"/World/cart/{name}", (dx, 0.0, -wd), axis="Y")
        if variant == "welded wheels":
            joint.CreateLowerLimitAttr(0.0)
            joint.CreateUpperLimitAttr(0.0)
        if variant in ("speed servos", "servo damping per radian"):
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(0.0)
            # USD reads the gains of angular drives per degree: N m s/deg
            per_degree = d.servo_gain * (1.0 if variant == "servo damping per radian"
                                         else math.pi / 180.0)
            drive.CreateDampingAttr(per_degree)
            drive.CreateMaxForceAttr(lim)
            drive.CreateTargetVelocityAttr(0.0)
    reach = _joint(stage, "prismatic", f"{J}/reach", "/World/cart/chassis", "/World/cart/tool",
                   (hl, 0.0, 0.0), axis="X")
    reach.CreateLowerLimitAttr(0.0)
    reach.CreateUpperLimitAttr(d.reach)
    drive = UsdPhysics.DriveAPI.Apply(reach.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(d.tool_kp)
    drive.CreateDampingAttr(d.tool_kv)
    drive.CreateMaxForceAttr(d.tool_force)
    drive.CreateTargetPositionAttr(0.0)

    # --- what does not touch what, as in Section 11.1 -----------------------------------
    UsdPhysics.FilteredPairsAPI.Apply(ground.GetPrim()).CreateFilteredPairsRel().SetTargets(
        [Sdf.Path("/World/cart/chassis"), Sdf.Path("/World/cart/tool")])
    UsdPhysics.FilteredPairsAPI.Apply(wall.GetPrim()).CreateFilteredPairsRel().SetTargets(
        [Sdf.Path("/World/cart/front"), Sdf.Path("/World/cart/rear")])
    del chassis, tool
    return {"root": "/World/cart", "ground": "/World/ground", "wall": "/World/wall",
            "scene": "/PhysicsScene", **{n: f"/World/cart/{n}" for n in LINKS}}
