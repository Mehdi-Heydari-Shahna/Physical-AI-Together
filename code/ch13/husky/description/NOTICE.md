# The Husky A300 description

The folders `clearpath_platform_description/` and `clearpath_control/` hold files copied, unchanged, from Clearpath Robotics' repository `clearpath_common`:

- Source: https://github.com/clearpathrobotics/clearpath_common, commit `78e3677412f1fed881f6436cdaf404f04f28ee4f` (19 October 2026), the release 2.9.18 of the packages `clearpath_platform_description` and `clearpath_control`.
- Copied: the URDF macros that assemble an A300 with outdoor wheels, the four-wheel differential drive and bumpers (`urdf/common.urdf.xacro`, `urdf/a300/a300.urdf.xacro`, `urdf/a300/drivetrain/*.urdf.xacro`, `urdf/a300/drivetrain/wheels/*.urdf.xacro`, `urdf/a300/attachments/bumper.urdf.xacro`, `urdf/generic/drivetrain/drivetrain.urdf.xacro`); the collision meshes of the chassis and the bumpers (`meshes/a300/chassis_collision.stl`, `meshes/a300/attachments/bumper_collision.stl`); the controller configuration of the A300's four-wheel differential drive (`clearpath_control/config/a300/control/diff_4wd.yaml`); and the two `package.xml` files, which give the release and the licence.
- Not copied: the visual meshes, which the simulators of Chapter 13 discard, and the collision meshes of the motors and suspension beams, links without mass that the model of Chapter 13 leaves out (see `../asset.py`).

The files are distributed under the BSD 3-Clause License of `clearpath_common`; a copy is in `LICENSE`. They are reproduced here under that licence so that the scripts of Chapter 13 run without ROS.

`husky_a300.urdf.xacro`, beside them, is ours: it calls Clearpath's macros with the options of Chapter 13. `husky_a300.urdf` is that file expanded by xacro 2.1.1 (`python3 ../asset.py --expand`); it is kept so that the scripts need no xacro.
