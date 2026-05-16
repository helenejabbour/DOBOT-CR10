# DOBOT-CR10

ROS2 trajectory planning and inverse dynamics computation for the Dobot CR10 robotic arm.

## Packages

- **`cr10_trajectories`** — Trajectory generation and visualization for CR10 robot
- **`cr10_torque_trajectories`** — Inverse dynamics computation using Pinocchio to compute required joint torques

## Prerequisites

- **ROS2 Humble** (or compatible distribution)
- **Python 3.10+**
- Ubuntu 22.04 LTS (WSL2 compatible)

## Installation

### 1. Install ROS2 dependencies

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

### 2. Clone and build

```bash
cd ~/ros2_ws
git clone https://github.com/<YOUR_USERNAME>/DOBOT-CR10.git src/DOBOT-CR10
cd ~/ros2_ws
colcon build
```

### 3. Source the workspace

```bash
source ~/ros2_ws/install/setup.bash
```

## Usage

### Launch CR10 with RViz visualization

```bash
ros2 launch cr10_trajectories cr10_rviz.launch.py
```

### Generate and execute trajectories

```bash
ros2 run cr10_trajectories trajectory_node
```

### Compute torque trajectories

```bash
ros2 run cr10_torque_trajectories torque_computation_node
```

## Project Structure

```
DOBOT-CR10/
├── src/
│   ├── cr10_trajectories/         # Trajectory planning package
│   │   ├── cr10_trajectories/     # Python module
│   │   ├── launch/                # ROS2 launch files
│   │   ├── setup.py
│   │   ├── package.xml
│   │   └── README.md
│   ├── cr10_torque_trajectories/  # Inverse dynamics package
│   │   ├── cr10_torque_trajectories/
│   │   ├── launch/
│   │   ├── setup.py
│   │   ├── package.xml
│   │   └── README.md
├── .gitignore
└── README.md
```

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `rclpy` | ROS2 Python client library |
| `numpy`, `scipy` | Numerical computing |
| `pinocchio` | Rigid body dynamics & inverse kinematics |
| `visualization_msgs`, `geometry_msgs` | RViz visualization |

## Documentation

See individual package READMEs:
- [cr10_trajectories/README.md](src/cr10_trajectories/README.md)
- [cr10_torque_trajectories/README.md](src/cr10_torque_trajectories/README.md)

## License

MIT
