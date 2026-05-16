# CR10 Torque Trajectory Computation — Part 2

Computes joint torque trajectories $\tau(t)$ for the Dobot CR10 from desired Cartesian end-effector paths. The full pipeline is:

$$\mathbf{x}_d(t),\,\dot{\mathbf{x}}_d(t),\,\ddot{\mathbf{x}}_d(t) \;\longrightarrow\; \mathbf{q}(t),\,\dot{\mathbf{q}}(t),\,\ddot{\mathbf{q}}(t) \;\longrightarrow\; \boldsymbol{\tau}(t)$$

Two trajectories are studied — straight-line and circular — each evaluated at two motion durations to characterize the effect of timing on actuator torque requirements.

---

## Pipeline

| Stage | What it does | Key implementation |
|---|---|---|
| **Part A** | Cartesian trajectory generation + IK | Quintic time scaling; nullspace-regularized DLS IK via Pinocchio |
| **Part B** | Joint velocity and acceleration | Translational Jacobian pseudoinverse; Jacobian time derivative in `LOCAL_WORLD_ALIGNED` |
| **Part C** | Torque computation | Pinocchio RNEA; RNEA boundary sanity check at $t=0$ and $t=T$ |

### Design decisions

- **Single Pinocchio model** throughout — no parallel symbolic model. FK, Jacobian, Jacobian time derivative, and RNEA all use the same URDF-based model, eliminating numerical mismatch between pipeline stages.
- **Position-only IK with nullspace regularization** — the 3-DOF task leaves a 3-DOF nullspace. A secondary joint-space objective pulls each solution toward the previous configuration, preventing wrist discontinuities without imposing an orientation target.
- **Damped least-squares (DLS)** — IK uses $\lambda = 10^{-2}$; Jacobian pseudoinverse uses $\lambda = 10^{-3}$. These are deliberately different: IK needs heavier regularization to converge stably across candidate seeds.
- **Single-pass velocity/acceleration loop** — $\dot{\mathbf{q}}_k$ is computed first, then immediately used to evaluate $\dot{J}_p$ for $\ddot{\mathbf{q}}_k$ in the same time step, ensuring frame consistency.
- **IK validation at 5 mm** — every accepted solution is checked against `IK_POSITION_TOL = 0.005 m` before entering the dynamics pipeline.
- **RNEA boundary check** — at $t=0$ and $t=T$ the quintic guarantees $\dot{\mathbf{q}} = \ddot{\mathbf{q}} = \mathbf{0}$, so $\boldsymbol{\tau} = \mathbf{g}(\mathbf{q})$ exactly. A warning is raised if this is violated beyond 0.5 N·m.

---

## Requirements

- ROS 2 (tested on Humble)
- `python3-pinocchio`
- `numpy`, `scipy`, `matplotlib`
- Dobot CR10 URDF (from `DOBOT_6Axis_ROS2_V4`)

---

## Build and run

```bash
cd ~/ros2_ws
colcon build --packages-select cr10_torque_trajectories
source install/setup.bash
ros2 run cr10_torque_trajectories compute_torque_trajectories
```

The URDF is located automatically. If auto-detection fails, pass it explicitly with `--urdf`.

### CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--durations` | `3.0 6.0` | Motion durations in seconds (minimum two required) |
| `--dt` | `0.01` | Time step [s] |
| `--damping` | `1e-3` | DLS damping factor for Jacobian pseudoinverse |
| `--urdf` | auto | Path to `cr10_robot.urdf` |
| `--ee-frame` | auto | Pinocchio end-effector frame name (default: `Link6`) |
| `--output-dir` | `torque_results` | Directory for all output files |
| `--line-a` | `0.45 -0.30 0.45` | Straight-line start point [m] |
| `--line-b` | `0.45 0.30 0.45` | Straight-line end point [m] |
| `--circle-center` | `0.50 0.00 0.45` | Circle center [m] |
| `--circle-radius` | `0.12` | Circle radius [m] |

Full example:

```bash
ros2 run cr10_torque_trajectories compute_torque_trajectories -- \
  --durations 3 6 \
  --dt 0.01 \
  --damping 0.001 \
  --urdf ~/ros2_ws/src/DOBOT_6Axis_ROS2_V4/dobot_rviz/urdf/cr10_robot.urdf \
  --ee-frame Link6 \
  --output-dir torque_results
```

The URDF can also be set via environment variable:

```bash
export CR10_URDF=~/ros2_ws/src/DOBOT_6Axis_ROS2_V4/dobot_rviz/urdf/cr10_robot.urdf
```

---

## Outputs

All outputs are written to `--output-dir` (default: `torque_results/`).

### Per trajectory and duration

| File | Contents |
|---|---|
| `*_data.npz` | `time, s, s_dot, s_ddot, x, x_dot, x_ddot, q, q_dot, q_ddot, tau` |
| `*_cartesian.png` | Cartesian position components + XY path projection |
| `*_q.png` | Joint positions $q_i(t)$ |
| `*_qdot.png` | Joint velocities $\dot{q}_i(t)$ |
| `*_qddot.png` | Joint accelerations $\ddot{q}_i(t)$ |
| `*_tau.png` | Joint torques $\tau_i(t)$ |

### Across durations

| File | Contents |
|---|---|
| `straight_torque_duration_comparison.png` | Torque overlay for all durations, straight-line |
| `circle_torque_duration_comparison.png` | Torque overlay for all durations, circular |

### Summary

`summary.json` contains:
- Pinocchio model info: `nq`, `nv`, joint order, end-effector frame
- Equations of motion terms and their physical meaning
- Per-joint peak values for $\dot{q}$, $\ddot{q}$, $\tau$ at each duration
- Quantitative answers to the five duration-study discussion questions
- Requirement coverage checklist for Parts A, B, and C

---

## Package structure

```
cr10_torque_trajectories/
├── torque_pipeline.py   # Main pipeline: trajectory gen, IK, Jacobians, RNEA, plots
├── ik_solver.py         # IK candidate search, scoring, warm-start strategy
├── ik_model.py          # Core DLS IK loop with nullspace regularization
└── __init__.py
```
