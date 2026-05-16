# DOBOT CR10 — ROS 2 Trajectory Planning and Inverse Dynamics

Full kinematic and dynamic pipeline for the Dobot CR10 6-DOF robotic arm, implemented from first principles in ROS 2.

---

## Packages

| Package | Description |
|---------|-------------|
| `cr10_trajectories` | Symbolic FK/IK, Cartesian trajectory generation, RViz visualization |
| `cr10_torque_trajectories` | Inverse dynamics via Pinocchio — computes required joint torques along trajectories |

---

## Prerequisites

- **ROS 2 Jazzy** (or compatible distribution)
- **Python 3.10+**
- Ubuntu 24.04 LTS (WSL2 compatible)

---

## Installation

### 1. Install system dependencies

```bash
sudo apt update
sudo apt install -y \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-numpy \
  python3-scipy \
  python3-matplotlib \
  pinocchio
```

### 2. Install Python dependencies

```bash
pip3 install sympy --break-system-packages
```

### 3. Clone and build

```bash
cd ~/ros2_ws/src
git clone https://github.com/<YOUR_USERNAME>/DOBOT-CR10.git
cd ~/ros2_ws
colcon build
```

### 4. Source the workspace

```bash
source ~/ros2_ws/install/setup.bash
```

Add to `~/.bashrc` to source automatically in every new terminal:

```bash
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
echo "source ~/ros2_ws/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

---

## Package 1 — `cr10_trajectories`

Trajectory generation, IK solving, and RViz visualization for the CR10.

### What is implemented

- **Symbolic forward kinematics** — DH chain T01 × … × T56 derived with sympy, compiled to numpy via lambdify
- **Geometric Jacobian** — position + angular Jacobian, compiled the same way
- **Damped Least Squares IK** — adaptive damping, joint limit enforcement, 1 mm position tolerance
- **Multi-seed IK branch search** — ranked by joint continuity and Yoshikawa manipulability
- **Straight-line trajectory** — Cartesian interpolation A → B with IK at every waypoint
- **Circular trajectory** — constant-Z parametric circle with IK at every waypoint
- **Real-time trace visualization** — LINE_STRIP marker on `/ee_trace`, waypoint dots on `/ee_trace_array`
- **tkinter GUI** — two-tab interface for live trajectory input with status feedback

### Run

```bash
# Terminal 1 — robot state publisher
ros2 launch dobot_rviz dobot_rviz.launch.py

# Terminal 2 — RViz
rviz2 -d ~/ros2_ws/cr10.rviz

# Terminal 3 — initialize joints to zero pose (exits automatically)
ros2 run cr10_trajectories initial_pose

# Terminal 4 — trajectory node
ros2 run cr10_trajectories trajectory_node

# Terminal 5 — GUI
ros2 run cr10_trajectories gui_node
```

See [`cr10_trajectories/README.md`](src/cr10_trajectories/README.md) for full details.

---

## Package 2 — `cr10_torque_trajectories`

Inverse dynamics computation for the CR10 using Pinocchio.

### What is implemented

- Joint torque computation along planned trajectories using the recursive Newton-Euler algorithm
- Pinocchio-based URDF loading and rigid body model construction
- Nullspace-regularized IK using the translational Jacobian pseudoinverse for position tracking without artificial orientation constraints

### Run

```bash
ros2 run cr10_torque_trajectories torque_computation_node
```

See [`cr10_torque_trajectories/README.md`](src/cr10_torque_trajectories/README.md) for full details.

---

## Project structure

```
DOBOT-CR10/
├── cr10_trajectories/
│   ├── cr10_trajectories/
│   │   ├── fk_model.py          # Symbolic FK + Jacobian (sympy + lambdify)
│   │   ├── ik_model.py          # Damped Least Squares IK solver
│   │   ├── ik_solver.py         # Multi-seed branch search + ROS integration
│   │   ├── trajectory_node.py   # ROS 2 joint state publisher + trajectory generation
│   │   ├── gui_node.py          # tkinter GUI
│   │   └── initial_pose.py      # Zero-pose initializer
│   ├── setup.py
│   ├── package.xml
│   └── README.md
├── cr10_torque_trajectories/
│   ├── cr10_torque_trajectories/
│   ├── setup.py
│   ├── package.xml
│   └── README.md
├── .gitignore
└── README.md
```

---

## Key dependencies

| Library | Purpose |
|---------|---------|
| `sympy` | Symbolic FK derivation and Jacobian computation |
| `numpy` | Numerical computation |
| `scipy` | Rotation utilities |
| `pinocchio` | Rigid body dynamics and inverse dynamics (`cr10_torque_trajectories`) |
| `rclpy` | ROS 2 Python client |
| `sensor_msgs`, `visualization_msgs`, `geometry_msgs`, `std_msgs` | ROS 2 message types |
| `tkinter` | GUI (standard Python library) |

No MoveIt. No full-stack solvers.

---

## Documentation

- [cr10_trajectories/README.md](src/cr10_trajectories/README.md)
- [cr10_torque_trajectories/README.md](src/cr10_torque_trajectories/README.md)
