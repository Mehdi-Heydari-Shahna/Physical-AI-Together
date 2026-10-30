# The UR5e description

The folder `ur_description/` holds files copied without change from Universal Robots' ROS 2 description of its arms:

- repository: https://github.com/UniversalRobots/Universal_Robots_ROS2_Description
- release 3.6.0, the head of the repository's branch `jazzy` (for the ROS 2 distribution Jazzy) on 24 October 2026, commit `7ce98d8c3512b9d0239da93b75de68ea64f61bd6` (22 October 2026)
- licence: BSD 3-Clause (`ur_description/LICENSE`; `package.xml` declares it for the package and for the UR5e's meshes)

Copied:

| File | Content |
|---|---|
| `package.xml` | the package's name, version and licences |
| `config/ur5e/default_kinematics.yaml` | the origins of the six joints (UR's nominal values, not a calibration of a particular arm) |
| `config/ur5e/physical_parameters.yaml` | the links' masses, centres of mass, inertia tensors and the meshes' offsets |
| `config/ur5e/joint_limits.yaml` | the joints' ranges, speeds and torques, with the sources that the file names |
| `config/ur5e/visual_parameters.yaml` | the meshes of each link and their offsets |
| `meshes/ur5e/collision/*.stl` | the collision meshes of the seven links |
| `urdf/ur_macro.xacro`, `urdf/inc/ur_common.xacro` | UR's macro, which turns the configuration files into a URDF; copied for reference only |

Not copied: the visual meshes, the other arms' files, the launch and test files.

Chapter 14 does not run xacro. The module `arm/model.py` reads the four configuration files and does the mapping of UR's macro itself, straight into MuJoCo's model: the chain of joints and links, each link's inertial, each collision mesh at the macro's placement, and the frames `flange` and `tool0`. Its docstring lists the mapping and what the model adds that the description does not hold.
