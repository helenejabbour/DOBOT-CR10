# CR10 IK and End-Effector Trajectory Control (ROS 2)

This package implements a full inverse-kinematics trajectory workflow for the Dobot CR10 in RViz without MoveIt.

## What is implemented

1. Kinematic model of CR10
- DH chain and symbolic forward kinematics in `cr10_trajectories/fk_model.py`
- Geometric Jacobian (position + orientation)

2. Inverse kinematics
- Damped Least Squares IK in `cr10_trajectories/ik_model.py`
- Candidate branch search + continuity scoring in `cr10_trajectories/ik_solver.py`
- FK validation of IK result (`|target - FK(q)|` tolerance check)

3. Custom ROS 2 joint-state control node
- `cr10_trajectories/trajectory_node.py` publishes `sensor_msgs/JointState` to `/joint_states`
- No Joint State Publisher GUI is required

4. RViz trajectory visualization
- End-effector trace published as `visualization_msgs/Marker` line strip on `/ee_trace`
- Per-waypoint dots published as `visualization_msgs/MarkerArray` on `/ee_trace_array`

5. Required motion functions
- Straight-line A -> B Cartesian interpolation + IK per waypoint
- Circle in plane parallel to ground (constant Z) around center A with radius R
- Fifth-order (quintic) time scaling for smooth start/stop motion profiles

## Topics and interfaces

Subscribed:
- `target_points` (`std_msgs/Float64MultiArray`):
  - `[Ax, Ay, Az, Bx, By, Bz, speed]`
  - optional 8th value for elbow mode (`-1` down, `0` auto, `1` up)
- `circle_params` (`std_msgs/Float64MultiArray`):
  - `[Cx, Cy, Cz, radius, steps]`
  - optional 6th value for elbow mode (`-1` down, `0` auto, `1` up)

Published:
- `/joint_states` (`sensor_msgs/JointState`)
- `/ee_trace` (`visualization_msgs/Marker`)
- `/ee_trace_array` (`visualization_msgs/MarkerArray`)
- `trajectory_status` (`std_msgs/Float64MultiArray`)

## Run

Build and source:

```bash
cd ~/ros2_ws
colcon build --packages-select cr10_trajectories
source install/setup.bash
```

Start trajectory controller + GUI:

```bash
ros2 launch cr10_trajectories cr10_ik_trajectory.launch.py
```

Optional: reset robot to zero joint pose:

```bash
ros2 run cr10_trajectories initial_pose
```

## RViz setup notes

- Fixed Frame: `base_link`
- Add robot model display for the CR10 URDF
- Add Marker display for topic `/ee_trace`
- Add MarkerArray display for topic `/ee_trace_array`

## Implementation notes

- Workspace reachability checks are applied before IK solving.
- A minimum Z constraint is applied to avoid floor collisions.
- Candidate IK solutions are optimized for continuity to reduce branch jumping.
