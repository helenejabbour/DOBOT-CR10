# CR10 Torque Trajectory Computation (Part 2)

This package computes CR10 joint torque trajectories from desired Cartesian end-effector trajectories using:

1. Quintic time-scaled Cartesian path generation (straight and circular).
2. Inverse kinematics for joint positions `q(t)`.
3. Pinocchio Jacobian and Jacobian time derivative for `q_dot(t)` and `q_ddot(t)`.
4. Pinocchio inverse dynamics (RNEA) for torques `tau(t)`.
5. Required plots for trajectory, joint states, torques, and duration comparison.

## Requirements coverage

The implementation in `cr10_torque_trajectories/torque_pipeline.py` covers:

- **Part A**: time vector, quintic `s/s_dot/s_ddot`, straight and circular Cartesian trajectories, IK-based `q(t)`.
- **Part B**: Pinocchio CR10 model loading, translational Jacobian `Jp`, Jacobian derivative `Jp_dot`, Jacobian-based `q_dot(t)` and `q_ddot(t)` in a consistent `LOCAL_WORLD_ALIGNED` frame.
- **Part C**: inverse dynamics torques with `pinocchio.rnea`, per-joint torque plots, and multi-duration comparison.

The run output writes:

- `*_data.npz` with `time, s, s_dot, s_ddot, x, x_dot, x_ddot, q, q_dot, q_ddot, tau`
- Required figures:
  - `*_cartesian.png`
  - `*_q.png`
  - `*_qdot.png`
  - `*_qddot.png`
  - `*_tau.png`
  - `straight_torque_duration_comparison.png`
  - `circle_torque_duration_comparison.png`
- `summary.json` containing:
  - detected Pinocchio model info (`nq`, `nv`, joint order, end-effector frame)
  - equations of motion + physical meaning of each term
  - duration-effect metrics and direct answers to the 5 required discussion questions
  - structured requirement coverage checklist

## Run

```bash
cd ~/ros2_ws
colcon build --packages-select cr10_torque_trajectories
source install/setup.bash
ros2 run cr10_torque_trajectories compute_torque_trajectories
```

## Optional arguments

```bash
ros2 run cr10_torque_trajectories compute_torque_trajectories -- \
  --durations 3 6 \
  --dt 0.01 \
  --damping 0.001 \
  --urdf ~/ros2_ws/src/DOBOT_6Axis_ROS2_V4/dobot_rviz/urdf/cr10_robot.urdf \
  --ee-frame Link6 \
  --output-dir torque_results
```

Outputs are saved to `torque_results/` by default.

> Note: the duration study enforces **at least two different durations** (e.g., `--durations 3 6`).
