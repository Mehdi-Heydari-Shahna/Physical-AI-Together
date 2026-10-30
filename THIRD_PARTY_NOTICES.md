# Third-party files

This repository contains files written by others, each under its own licence, which the licence of the rest of the repository does not change. Each folder below has a `NOTICE.md` with the source, the commit and the list of files copied, and a copy of the licence.

| Folder or file | What it is | Source | Licence |
|---|---|---|---|
| `code/ch13/husky/description/` | URDF macros, two collision meshes, the controller configuration and the package files of the Husky A300, copied unchanged | Clearpath Robotics, [clearpath_common](https://github.com/clearpathrobotics/clearpath_common), packages `clearpath_platform_description` and `clearpath_control` 2.9.18, commit `78e3677412f1fed881f6436cdaf404f04f28ee4f` | BSD 3-Clause (`LICENSE` in the folder) |
| `code/ch13/husky/drive.py` | A translation into Python of parts of `diff_drive_controller` and of the rate limiter of `control_toolbox` | [ros2_controllers](https://github.com/ros-controls/ros2_controllers), branch `jazzy`, commit `d71703b`; [control_toolbox](https://github.com/ros-controls/control_toolbox), branch `jazzy`, commit `1e54022`; "Copyright 2020 PAL Robotics S.L." | Apache License 2.0 (`code/ch13/husky/LICENSE-Apache-2.0.txt`) |
| `code/ch14/arm/description/ur_description/` | Configuration files, collision meshes and the macro of the UR5e, copied unchanged | Universal Robots, [Universal_Robots_ROS2_Description](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description), release 3.6.0, commit `7ce98d8c3512b9d0239da93b75de68ea64f61bd6` | BSD 3-Clause (`LICENSE` in the folder) |
| `code/ch15/quad/description/` | The model of the Go2 (`go2.xml`), its README and change log, copied unchanged | [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie), folder `unitree_go2/`, commit `f3b6013629c7b94f8ee93fd977b161fa8350f9bc`; copyright Unitree Robotics | BSD 3-Clause (`LICENSE` in the folder) |
| `code/ch10/providers/robocompiler-stewart/` | Three JSON files written, unchanged, by RoboCompiler's compile step for its six-UPS Stewart platform | [RoboCompiler](https://github.com/Mehdi-Heydari-Shahna/RoboCompiler), commit `a4e4437c18cbc4485b346e71674a9d26cd261be4`, by Mehdi Heydari Shahna, Joongheon Kim and Jouni Mattila | Apache License 2.0 (`LICENSE` in the folder) |

## Used from checkouts, not copied

Some scripts read files from checkouts of other repositories, which are not part of this repository and keep their own terms: the visual meshes of the Husky A300 (clearpath_common), of the UR5e (Universal_Robots_ROS2_Description) and of the Go2 (MuJoCo Menagerie), which the videos and some stills draw; and the package `teamimp` of [Team-Impedance-Control](https://github.com/Mehdi-Heydari-Shahna/Team-Impedance-Control), with its copy of MuJoCo Menagerie's Apptronik Apollo, which Chapter 15's humanoid run loads; and, for the published learners of `code/ch06/published`, [TD-MPC2](https://github.com/nicklashansen/tdmpc2) at commit `332f19e4aba4bffe2dc736f45ea7f7bdc585903a` and [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) at commit `5ba07ac6661db573af695b419a7947ecb704690f`, with Stable-Baselines3 and the other packages that its `requirements-*.txt` files pin, installed with pip. `code/render/README.md` and each chapter's README give the commits. The scene bundles that `code/render/cycles` writes contain such meshes and are for rendering on one's own machine, not for distribution.

## Logs of other software

`code/ch11/results/isaac-logs/` and the console files under `code/ch13/results/isaac/` hold what Isaac Sim printed during the book's runs, and the `console.txt` files under `code/ch06/published/results/` what the published learners' libraries printed; they are kept as the record of those runs.
