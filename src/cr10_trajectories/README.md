# CR10 Inverse Kinematics and Trajectory Control — ROS 2

End-effector trajectory control for the Dobot CR10 in RViz, implemented from first principles without MoveIt.

---

## Overview

This package implements a complete IK-based trajectory workflow for the Dobot CR10 6-DOF manipulator:

- Symbolic forward kinematics and geometric Jacobian derived from DH parameters
- Damped Least Squares IK solver with multi-seed branch search and continuity scoring
- Custom ROS 2 node that publishes joint states programmatically (no GUI sliders)
- Real-time end-effector trace visualization in RViz
- Straight-line and circular trajectory generation with IK at every waypoint
- tkinter GUI for live trajectory input

---

## Package structure

```
cr10_trajectories/
├── fk_model.py          # Symbolic FK chain + geometric Jacobian (sympy + lambdify)
├── ik_model.py          # Damped Least Squares IK solver
├── ik_solver.py         # Multi-seed branch search, manipulability scoring, ROS integration layer
├── trajectory_node.py   # ROS 2 node: joint state publisher + trace markers + trajectory generation
├── gui_node.py          # tkinter GUI: straight-line and circular trajectory input
└── initial_pose.py      # One-shot zero-pose initializer
```

---

## Kinematic model

**DH parameters (Dobot CR10, meters):**

| Joint | θ offset | d (m) | a (m) | α (rad) |
|-------|----------|--------|--------|---------|
| 1 | — | 0.1765 | 0 | π/2 |
| 2 | +π/2 | 0 | 0.607 | 0 |
| 3 | — | 0 | 0.568 | 0 |
| 4 | −π/2 | 0.193 | 0 | −π/2 |
| 5 | — | 0.125 | 0 | π/2 |
| 6 | — | 0.1114 | 0 | 0 |

The full FK chain T01 × T12 × … × T56 = T06 is derived symbolically using sympy, then compiled to numpy via `lambdify` for fast repeated evaluation during trajectory planning.

The geometric Jacobian (6×6) is derived symbolically and compiled the same way — computed once at startup, evaluated as pure numpy at runtime.

**IK solver** (`ik_model.py`): Iterative Damped Least Squares (Levenberg-Marquardt) with adaptive damping, joint limit enforcement (±160° on joint 3, ±360° on others), and position convergence tolerance of 1 mm.

**IK integration layer** (`ik_solver.py`): Multi-seed branch search across shoulder/elbow/wrist configurations, ranked by:
- Continuity to previous configuration (primary criterion)
- Yoshikawa position manipulability (singularity avoidance)

FK validation is applied to every IK result — solutions with position error > 5 mm are rejected.

---

## ROS 2 topics

| Direction | Topic | Message type | Content |
|-----------|-------|-------------|---------|
| Subscribed | `target_points` | `std_msgs/Float64MultiArray` | `[Ax, Ay, Az, Bx, By, Bz, speed]` |
| Subscribed | `circle_params` | `std_msgs/Float64MultiArray` | `[Cx, Cy, Cz, radius, steps]` |
| Published | `/joint_states` | `sensor_msgs/JointState` | 6 joint angles at 10 Hz |
| Published | `/ee_trace` | `visualization_msgs/Marker` | LINE_STRIP end-effector trace |
| Published | `/ee_trace_array` | `visualization_msgs/MarkerArray` | Per-waypoint sphere markers |
| Published | `trajectory_status` | `std_msgs/Float64MultiArray` | Status code (0=info, 1=warn, 2=error, 3=ok) |

---

## Build and run

```bash
cd ~/ros2_ws
colcon build --packages-select cr10_trajectories
source install/setup.bash
```

Run in separate terminals **in this order:**

```bash
# Terminal 1 — robot state publisher
ros2 launch dobot_rviz dobot_rviz.launch.py

# Terminal 2 — RViz
rviz2 -d ~/ros2_ws/cr10.rviz

# Terminal 3 — initialize joints to zero pose
ros2 run cr10_trajectories initial_pose

# Terminal 4 — trajectory node (wait for Terminal 3 to finish first)
ros2 run cr10_trajectories trajectory_node

# Terminal 5 — GUI
ros2 run cr10_trajectories gui_node
```

> **Note:** Terminal 3 (`initial_pose`) exits automatically after ~2.5 seconds. Launch Terminal 4 after it shuts down.

---

## RViz configuration

- **Fixed Frame:** `base_link`
- **RobotModel** display → topic `/robot_description`
- **Marker** display → topic `/ee_trace` (red trace line)
- **MarkerArray** display → topic `/ee_trace_array` (green waypoint dots)

A pre-configured `cr10.rviz` file is included at `~/ros2_ws/cr10.rviz`.

---

## GUI usage

The GUI has two tabs:

**Straight-Line (A → B)**
- Point A: x, y, z in meters
- Point B: x, y, z in meters
- Speed: 1 (slow) to 10 (fast)

**Circular Trajectory**
- Center: x, y, z in meters
- Radius: meters
- Speed: 1 (slow) to 10 (fast)

The status bar at the bottom of the GUI shows real-time feedback — trajectory ready, errors, and completion messages.

---

## Workspace constraints

- Reachability: `0.2 m < ‖p‖ < 1.3 m` from base
- Minimum Z: `0.15 m` (floor clearance)
- If any waypoint along a trajectory fails validation or IK, the **entire trajectory is aborted** — no partial drawing

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `sympy` | Symbolic FK and Jacobian derivation |
| `numpy` | Numerical computation |
| `scipy` | Rotation utilities (`spatial.transform`) |
| `rclpy` | ROS 2 Python client |
| `sensor_msgs`, `visualization_msgs`, `geometry_msgs`, `std_msgs` | ROS 2 message types |
| `tkinter` | GUI (standard Python library) |

No MoveIt. No ikpy. No full-stack solvers.
