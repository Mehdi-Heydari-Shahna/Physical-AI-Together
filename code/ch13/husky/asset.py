"""
The Husky A300 as Clearpath describes it, and as Chapter 13 uses it.

Physical AI, Together, Chapter 13 (Section 13.1).

The description is Clearpath's own (description/, see its NOTICE.md): the
URDF macros of clearpath_platform_description 2.9.18, assembled by
description/husky_a300.urdf.xacro and expanded by xacro into
description/husky_a300.urdf. This module

  * expands the macros again, without ROS (python3 asset.py --expand, with
    xacro installed);
  * gives the URDF text that the simulators load: the visual elements and
    the collision meshes of the links without mass (the motors and the
    suspension beams) removed, the paths of the meshes made relative, and,
    unless the published description is asked for, one correction (below);
  * reads the controller configuration of the A300's four-wheel
    differential drive (description/clearpath_control/.../diff_4wd.yaml);
  * and states the facts of the description that the record quotes.

The correction. In the published description, the joints that carry the
wheels on the suspension beams sit 9.5 mm to the robot's left of each beam
on both sides (suspension_beam.urdf.xacro gives the same origin, y = 0.0095,
to the left and to the right beam, whose frames are not mirrored). The left
wheels are then 0.2829 m from the centre line and the right wheels 0.2639 m:
a track of 0.5468 m, against the track width of 566 mm in the A300's user
manual. Mirroring the offset of the right beam gives 0.2829 m on both sides,
a track of 0.5658 m. The model of Chapter 13 makes that correction and
declares it; published=True gives the description as published.
"""
import argparse
import os
import re
import sys
import types
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
DESCRIPTION = os.path.join(HERE, "description")
PACKAGE = os.path.join(DESCRIPTION, "clearpath_platform_description")
XACRO = os.path.join(DESCRIPTION, "husky_a300.urdf.xacro")
URDF = os.path.join(DESCRIPTION, "husky_a300.urdf")
CONTROL = os.path.join(DESCRIPTION, "clearpath_control", "config", "a300", "control",
                       "diff_4wd.yaml")

SOURCE = ("clearpath_common, https://github.com/clearpathrobotics/clearpath_common, "
          "commit 78e3677412f1fed881f6436cdaf404f04f28ee4f (19 October 2026), "
          "clearpath_platform_description and clearpath_control 2.9.18, BSD 3-Clause")
OPTIONS = "a300 control=diff_4wd front_wheels=outdoor rear_wheels=outdoor, front and rear bumpers"
# links without mass whose collision meshes are not copied (NOTICE.md)
MASSLESS = re.compile(r"(_motor_link|_suspension_beam_link|_suspension_beam_spacer_link)$")
# the correction: the origin of the right beam's wheel mounts, mirrored
CORRECTED = {"right_suspension_beam_front_mount_joint": "0.256 -0.0095 -0.0085",
             "right_suspension_beam_rear_mount_joint": "-0.256 -0.0095 -0.0085"}
PUBLISHED = {"right_suspension_beam_front_mount_joint": "0.256 0.0095 -0.0085",
             "right_suspension_beam_rear_mount_joint": "-0.256 0.0095 -0.0085"}


# ----------------------------------------------------------------------------
# Expansion with xacro, without ROS
# ----------------------------------------------------------------------------
def expand(out=URDF):
    """Expands description/husky_a300.urdf.xacro into the URDF, resolving
    $(find clearpath_platform_description) to description/ through a stand-in
    for ament_index_python, which xacro uses for $(find)."""
    packages = types.ModuleType("ament_index_python.packages")
    packages.get_package_share_directory = lambda name: os.path.join(DESCRIPTION, name)
    index = types.ModuleType("ament_index_python")
    index.packages = packages
    sys.modules.setdefault("ament_index_python", index)
    sys.modules.setdefault("ament_index_python.packages", packages)
    import importlib.metadata                                        # noqa: E402
    import xacro                                                     # noqa: E402
    doc = xacro.process_file(XACRO, mappings={"use_platform_controllers": "false"})
    text = doc.toprettyxml(indent="  ")
    # xacro writes a banner with the absolute path of its input; keep the
    # file free of the paths of the machine that expanded it
    text = re.sub(r"<!-- (=+|\|[^\n]*\|) -->\n", "", text)
    release = importlib.metadata.version("xacro")
    text = text.replace('<?xml version="1.0" ?>',
                        '<?xml version="1.0" ?>\n<!-- Expanded by xacro ' + release +
                        " from husky_a300.urdf.xacro, with the option expand of "
                        "husky/asset.py. -->", 1)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    return out


# ----------------------------------------------------------------------------
# The URDF that the simulators load
# ----------------------------------------------------------------------------
def urdf_text(published=False, meshes="package"):
    """The URDF of the Husky as the simulators of Chapter 13 load it.

    published: the description as published (asymmetric track), not the
    declared correction. meshes: "package" leaves the paths of the meshes
    relative to the package folder (for MuJoCo's meshdir); "absolute" makes
    them absolute file paths (for Isaac Sim's URDF importer)."""
    tree = ET.parse(URDF)
    robot = tree.getroot()
    for link in robot.findall("link"):
        for visual in link.findall("visual"):
            link.remove(visual)
        if MASSLESS.search(link.get("name", "")):
            for collision in link.findall("collision"):
                link.remove(collision)
    origins = PUBLISHED if published else CORRECTED
    for joint in robot.findall("joint"):
        if joint.get("name") in origins:
            joint.find("origin").set("xyz", origins[joint.get("name")])
    for material in robot.findall("material"):
        robot.remove(material)
    text = ET.tostring(robot, encoding="unicode")
    prefix = "package://clearpath_platform_description/"
    if meshes == "absolute":
        text = text.replace(prefix, PACKAGE.replace(os.sep, "/") + "/")
    else:
        text = text.replace(prefix, "")
    return text


def write_urdf(path, published=False, meshes="absolute"):
    """Writes the URDF that urdf_text gives, for an importer that reads files."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write('<?xml version="1.0"?>\n' + urdf_text(published, meshes) + "\n")
    return path


# ----------------------------------------------------------------------------
# The controller configuration
# ----------------------------------------------------------------------------
def controller():
    """The parameters of platform_velocity_controller and the update rate of
    the controller manager in diff_4wd.yaml, read without a YAML library (the
    file is flat key: value lines under two levels of headings)."""
    out, section = {}, None
    with open(CONTROL, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.split("#", 1)[0].rstrip()
            if not line.strip():
                continue
            depth = len(line) - len(line.lstrip())
            key, _, value = line.strip().partition(":")
            if depth == 0:
                section = key
                continue
            value = value.strip()
            if not value:
                continue
            if section in ("platform_velocity_controller", "controller_manager"):
                out[key] = value
    return out


def number(value):
    v = value.strip()
    if v.lower() in (".nan", "nan"):
        return float("nan")
    return float(v)


# ----------------------------------------------------------------------------
# The facts of the description
# ----------------------------------------------------------------------------
def description_values():
    """The values that the macros set (read from the expanded URDF and the
    xacro properties), as the record quotes them."""
    robot = ET.parse(URDF).getroot()
    masses = {link.get("name"): float(link.find("inertial/mass").get("value"))
              for link in robot.findall("link") if link.find("inertial/mass") is not None}
    wheel = robot.find("link[@name='front_left_wheel_link']/collision/geometry/cylinder")
    text = open(os.path.join(PACKAGE, "urdf", "a300", "a300.urdf.xacro"), encoding="utf-8").read()
    prop = {m.group(1): m.group(2) for m in re.finditer(
        r'<xacro:property name="(\w+)" value="([^"$]+)"', text)}
    return dict(masses=masses, wheel_radius=float(wheel.get("radius")),
                wheel_width=float(wheel.get("length")),
                chassis_size=(float(prop["chassis_x_size"]), float(prop["chassis_y_size"]),
                              float(prop["chassis_z_size"])),
                wheel_base_property=float(prop["wheel_base"]),
                wheel_track_property=float(prop["wheel_track"]))


def main():
    ap = argparse.ArgumentParser(description="The Husky A300 description of Chapter 13")
    ap.add_argument("--expand", action="store_true",
                    help="expand husky_a300.urdf.xacro again (needs xacro)")
    args = ap.parse_args()
    if args.expand:
        print("written", os.path.relpath(expand(), HERE))
    v = description_values()
    print("masses:", v["masses"])
    print("wheel radius, width:", v["wheel_radius"], v["wheel_width"])
    print("controller:", controller())


if __name__ == "__main__":
    main()
