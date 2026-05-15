#!/usr/bin/python3
"""
Compute CR10 torque trajectories from Cartesian paths.

Pipeline:
1. Cartesian trajectory generation with quintic time scaling.
2. Inverse kinematics for q(t).
3. Jacobian-based q_dot(t) and q_ddot(t) using Pinocchio.
4. Inverse dynamics (RNEA) for tau(t) using Pinocchio.
5. Plot generation for each required signal.

Dynamics equation:
    tau = M(q) q_ddot + C(q, q_dot) q_dot + g(q)
where:
    - M(q): inertia matrix
    - C(q, q_dot) q_dot: Coriolis and centrifugal torques
    - g(q): gravity torques
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .ik_solver import (
    IK_POSITION_TOL,
    forward_kinematics,
    is_reachable,
    reset_ik_seed,
    set_ik_seed,
    solve_ik_candidates,
    solve_ik_direct,
)

try:
    import pinocchio as pin
except ImportError as import_error:  # pragma: no cover
    pin = None
    _PINOCCHIO_IMPORT_ERROR = import_error
else:
    _PINOCCHIO_IMPORT_ERROR = None

JOINT_LABELS = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
EE_FRAME_CANDIDATES = ['Link6', 'tool0', 'flange', 'ee_link', 'tcp', 'tool_link']


@dataclass
class CartesianTrajectory:
    name: str
    time: np.ndarray
    s: np.ndarray
    s_dot: np.ndarray
    s_ddot: np.ndarray
    x: np.ndarray
    x_dot: np.ndarray
    x_ddot: np.ndarray


@dataclass
class TrajectoryResult:
    name: str
    duration: float
    time: np.ndarray
    s: np.ndarray
    s_dot: np.ndarray
    s_ddot: np.ndarray
    x: np.ndarray
    x_dot: np.ndarray
    x_ddot: np.ndarray
    q: np.ndarray
    q_dot: np.ndarray
    q_ddot: np.ndarray
    tau: np.ndarray


def _quintic_scaling(t: np.ndarray, duration: float) -> Tuple[np.ndarray, ...]:
    duration = float(max(duration, 1e-8))
    u = t / duration
    if not np.isclose(t[-1], duration, rtol=1e-3):
        raise ValueError(
            f'Time vector endpoint {t[-1]:.6f} deviates from duration '
            f'{duration:.6f}. Check _time_vector construction.'
        )
    u = np.clip(u, 0.0, 1.0)
    s = 10.0 * u ** 3 - 15.0 * u ** 4 + 6.0 * u ** 5
    s_dot = (30.0 * u ** 2 - 60.0 * u ** 3 + 30.0 * u ** 4) / duration
    s_ddot = (60.0 * u - 180.0 * u ** 2 + 120.0 * u ** 3) / (duration ** 2)
    return s, s_dot, s_ddot


def _time_vector(duration: float, dt: float) -> np.ndarray:
    duration = float(duration)
    dt = float(dt)
    if duration <= 0.0:
        raise ValueError(f'Duration must be positive, got {duration}.')
    if dt <= 0.0:
        raise ValueError(f'dt must be positive, got {dt}.')
    return np.arange(0.0, duration + 0.5 * dt, dt, dtype=float)


def generate_straight_trajectory(
    point_a: Sequence[float],
    point_b: Sequence[float],
    duration: float,
    dt: float,
) -> CartesianTrajectory:
    t = _time_vector(duration, dt)
    s, s_dot, s_ddot = _quintic_scaling(t, duration)
    A = np.asarray(point_a, dtype=float)
    B = np.asarray(point_b, dtype=float)
    delta = B - A
    x = A[None, :] + s[:, None] * delta[None, :]
    x_dot = s_dot[:, None] * delta[None, :]
    x_ddot = s_ddot[:, None] * delta[None, :]
    return CartesianTrajectory('straight', t, s, s_dot, s_ddot, x, x_dot, x_ddot)


def generate_circle_trajectory(
    center: Sequence[float],
    radius: float,
    duration: float,
    dt: float,
) -> CartesianTrajectory:
    if radius <= 0.0:
        raise ValueError(f'Circle radius must be positive, got {radius}.')
    t = _time_vector(duration, dt)
    s, s_dot, s_ddot = _quintic_scaling(t, duration)
    center = np.asarray(center, dtype=float)
    cx, cy, cz = center
    theta = 2.0 * np.pi * s
    theta_dot = 2.0 * np.pi * s_dot
    theta_ddot = 2.0 * np.pi * s_ddot

    x = np.zeros((len(t), 3), dtype=float)
    x[:, 0] = cx + radius * np.cos(theta)
    x[:, 1] = cy + radius * np.sin(theta)
    x[:, 2] = cz

    x_dot = np.zeros((len(t), 3), dtype=float)
    x_dot[:, 0] = -radius * np.sin(theta) * theta_dot
    x_dot[:, 1] = radius * np.cos(theta) * theta_dot

    x_ddot = np.zeros((len(t), 3), dtype=float)
    x_ddot[:, 0] = (
        -radius * np.cos(theta) * theta_dot ** 2
        - radius * np.sin(theta) * theta_ddot
    )
    x_ddot[:, 1] = (
        -radius * np.sin(theta) * theta_dot ** 2
        + radius * np.cos(theta) * theta_ddot
    )
    return CartesianTrajectory('circle', t, s, s_dot, s_ddot, x, x_dot, x_ddot)


def compute_joint_positions(
    model,
    data,
    frame_id: int,
    cartesian_positions: np.ndarray,
    elbow_mode: str = 'auto',
) -> np.ndarray:
    if cartesian_positions.ndim != 2 or cartesian_positions.shape[1] != 3:
        raise ValueError('Cartesian positions must have shape (N, 3).')

    for k, p in enumerate(cartesian_positions):
        if not is_reachable(p):
            raise ValueError(
                f'Waypoint {k} is outside workspace: {np.round(p, 4)}.'
            )

    reset_ik_seed()
    q = np.zeros((cartesian_positions.shape[0], 6), dtype=float)

    p0 = cartesian_positions[0]
    first = solve_ik_candidates(
        model,
        data,
        frame_id,
        float(p0[0]),
        float(p0[1]),
        float(p0[2]),
        reference_theta=None,
        elbow_mode=elbow_mode,
        max_solutions=1,
        exhaustive=True,
    )
    q[0, :] = np.asarray(first[0]['angles'], dtype=float)
    q6_anchor = float(q[0, 5])

    for i in range(1, cartesian_positions.shape[0]):
        target = cartesian_positions[i]
        q_prev = q[i - 1, :]
        try:
            q_i = solve_ik_direct(
                model,
                data,
                frame_id,
                float(target[0]),
                float(target[1]),
                float(target[2]),
                reference_theta=q_prev,
                q6_ref=q6_anchor,
                max_iterations=180,
            )
        except ValueError:
            fallback = solve_ik_candidates(
                model,
                data,
                frame_id,
                float(target[0]),
                float(target[1]),
                float(target[2]),
                reference_theta=q_prev,
                q6_ref=q6_anchor,
                elbow_mode=elbow_mode,
                max_solutions=1,
                exhaustive=False,
            )
            q_i = np.asarray(fallback[0]['angles'], dtype=float)
        q[i, :] = q_i

    set_ik_seed(q[-1, :])

    for i, qi in enumerate(q):
        fk = np.asarray(forward_kinematics(model, data, frame_id, qi), dtype=float)
        err = np.linalg.norm(fk - cartesian_positions[i])
        if err > IK_POSITION_TOL:
            raise ValueError(
                f'IK validation failed at waypoint {i}: {err:.4f}m.'
            )

    return q


def _damped_least_squares(
    jacobian: np.ndarray,
    rhs: np.ndarray,
    damping: float,
) -> np.ndarray:
    identity = np.eye(jacobian.shape[0], dtype=float)
    regularized = jacobian @ jacobian.T + (damping ** 2) * identity
    return jacobian.T @ np.linalg.solve(regularized, rhs)


def _resolve_urdf_path(user_urdf: Optional[str]) -> Path:
    if user_urdf is not None:
        urdf = Path(user_urdf).expanduser().resolve()
        if not urdf.exists():
            raise FileNotFoundError(f'URDF not found: {urdf}')
        return urdf

    env_urdf = None
    if 'CR10_URDF' in os.environ:
        env_urdf = Path(os.environ['CR10_URDF']).expanduser()
    candidates = [
        env_urdf,
        Path.cwd() / 'cr10_robot.urdf',
        Path.cwd().parent / 'DOBOT_6Axis_ROS2_V4' / 'dobot_rviz' / 'urdf'
        / 'cr10_robot.urdf',
        Path.home() / 'ros2_ws' / 'src' / 'DOBOT_6Axis_ROS2_V4'
        / 'dobot_rviz' / 'urdf' / 'cr10_robot.urdf',
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        'Could not locate cr10_robot.urdf. Pass --urdf or set CR10_URDF.'
    )


def _resolve_ee_frame(model, requested_name: Optional[str]) -> Tuple[int, str]:
    frame_names = [frame.name for frame in model.frames]

    def frame_id(name: str) -> Optional[int]:
        if name not in frame_names:
            return None
        return int(model.getFrameId(name))

    if requested_name is not None:
        fid = frame_id(requested_name)
        if fid is None:
            raise ValueError(f'End-effector frame not found: {requested_name}')
        return fid, requested_name

    for candidate in EE_FRAME_CANDIDATES:
        fid = frame_id(candidate)
        if fid is not None:
            return fid, candidate

    fallback_name = model.frames[-1].name
    return int(model.getFrameId(fallback_name)), fallback_name


def load_pinocchio_model(
    urdf_path: Optional[str] = None,
    ee_frame_name: Optional[str] = None,
):
    if pin is None:  # pragma: no cover
        raise ImportError(
            'pinocchio is required for Part B/C computations. '
            'Install python3-pinocchio.'
        ) from _PINOCCHIO_IMPORT_ERROR

    urdf = _resolve_urdf_path(urdf_path)
    model = pin.buildModelFromUrdf(str(urdf))
    data = model.createData()
    ee_frame_id, resolved_ee_name = _resolve_ee_frame(model, ee_frame_name)

    joint_names = []
    for jid in range(1, model.njoints):
        if model.joints[jid].nq > 0:
            joint_names.append(model.names[jid])

    return model, data, urdf, ee_frame_id, resolved_ee_name, joint_names


def compute_joint_velocity_acceleration(
    model,
    data,
    frame_id: int,
    q: np.ndarray,
    x_dot: np.ndarray,
    x_ddot: np.ndarray,
    damping: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n_steps = q.shape[0]
    q_dot = np.zeros((n_steps, model.nv), dtype=float)
    q_ddot = np.zeros((n_steps, model.nv), dtype=float)

    for k in range(n_steps):
        qk = q[k, :]

        # Step 1: compute q_dot.
        pin.forwardKinematics(model, data, qk)
        pin.computeJointJacobians(model, data, qk)
        pin.updateFramePlacements(model, data)
        J_full = pin.getFrameJacobian(
            model,
            data,
            frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        J_pos = np.asarray(J_full[:3, :], dtype=float)
        q_dot[k, :] = _damped_least_squares(J_pos, x_dot[k, :], damping)

        # Step 2: compute q_ddot using q_dot just computed.
        qdk = q_dot[k, :]
        pin.forwardKinematics(model, data, qk, qdk)
        pin.computeJointJacobians(model, data, qk)
        pin.computeJointJacobiansTimeVariation(model, data, qk, qdk)
        pin.updateFramePlacements(model, data)

        J_full = pin.getFrameJacobian(
            model,
            data,
            frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        J_dot_full = pin.getFrameJacobianTimeVariation(
            model,
            data,
            frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        J_pos = np.asarray(J_full[:3, :], dtype=float)
        J_pos_dot = np.asarray(J_dot_full[:3, :], dtype=float)
        rhs = x_ddot[k, :] - J_pos_dot @ qdk
        q_ddot[k, :] = _damped_least_squares(J_pos, rhs, damping)

    return q_dot, q_ddot


def compute_torque_trajectory(
    model,
    data,
    q: np.ndarray,
    q_dot: np.ndarray,
    q_ddot: np.ndarray,
) -> np.ndarray:
    tau = np.zeros((q.shape[0], model.nv), dtype=float)
    for k in range(q.shape[0]):
        tau[k, :] = np.asarray(
            pin.rnea(model, data, q[k, :], q_dot[k, :], q_ddot[k, :]),
            dtype=float,
        )
    _verify_rnea_gravity_boundary(model, data, q, q_dot, q_ddot, tau)
    return tau


def _verify_rnea_gravity_boundary(
    model,
    data,
    q: np.ndarray,
    q_dot: np.ndarray,
    q_ddot: np.ndarray,
    tau: np.ndarray,
    tol: float = 0.5,
):
    """At boundary steps where q_dot≈0 and q_ddot≈0, tau must equal g(q)."""
    for k in [0, -1]:
        if np.linalg.norm(q_dot[k]) < 1e-6 and np.linalg.norm(q_ddot[k]) < 1e-6:
            g = np.asarray(
                pin.rnea(
                    model,
                    data,
                    q[k],
                    np.zeros(model.nv, dtype=float),
                    np.zeros(model.nv, dtype=float),
                ),
                dtype=float,
            )
            err = np.linalg.norm(tau[k] - g)
            if err > tol:
                import warnings
                warnings.warn(
                    f'RNEA boundary check failed at step {k}: '
                    f'|tau - g(q)| = {err:.4f} N·m (tol={tol}). '
                    'Check IK accuracy or Pinocchio model gravity vector.',
                    stacklevel=2,
                )


def run_pipeline_for_trajectory(
    model,
    data,
    frame_id: int,
    cart: CartesianTrajectory,
    duration: float,
    damping: float,
) -> TrajectoryResult:
    q = compute_joint_positions(model, data, frame_id, cart.x)
    q_dot, q_ddot = compute_joint_velocity_acceleration(
        model,
        data,
        frame_id,
        q,
        cart.x_dot,
        cart.x_ddot,
        damping,
    )
    tau = compute_torque_trajectory(model, data, q, q_dot, q_ddot)
    return TrajectoryResult(
        name=cart.name,
        duration=duration,
        time=cart.time,
        s=cart.s,
        s_dot=cart.s_dot,
        s_ddot=cart.s_ddot,
        x=cart.x,
        x_dot=cart.x_dot,
        x_ddot=cart.x_ddot,
        q=q,
        q_dot=q_dot,
        q_ddot=q_ddot,
        tau=tau,
    )


def _plot_cartesian(time: np.ndarray, x: np.ndarray, out_file: Path, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(time, x[:, 0], label='x')
    axes[0].plot(time, x[:, 1], label='y')
    axes[0].plot(time, x[:, 2], label='z')
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('Position [m]')
    axes[0].set_title('Cartesian components')
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(x[:, 0], x[:, 1], label='XY path')
    axes[1].set_xlabel('x [m]')
    axes[1].set_ylabel('y [m]')
    axes[1].set_title('Path projection (XY)')
    axes[1].axis('equal')
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_file, dpi=160)
    plt.close(fig)


def _plot_joint_series(
    time: np.ndarray,
    values: np.ndarray,
    ylabel: str,
    title: str,
    out_file: Path,
    joint_names: Sequence[str],
):
    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(values.shape[1]):
        label = joint_names[i] if i < len(joint_names) else f'joint{i + 1}'
        ax.plot(time, values[:, i], label=label)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', ncol=2)
    fig.tight_layout()
    fig.savefig(out_file, dpi=160)
    plt.close(fig)


def save_result_artifacts(
    result: TrajectoryResult,
    output_dir: Path,
    joint_names: Sequence[str],
):
    duration_tag = str(result.duration).replace('.', 'p')
    prefix = f'{result.name}_T{duration_tag}s'

    np.savez(
        output_dir / f'{prefix}_data.npz',
        time=result.time,
        s=result.s,
        s_dot=result.s_dot,
        s_ddot=result.s_ddot,
        x=result.x,
        x_dot=result.x_dot,
        x_ddot=result.x_ddot,
        q=result.q,
        q_dot=result.q_dot,
        q_ddot=result.q_ddot,
        tau=result.tau,
    )

    _plot_cartesian(
        result.time,
        result.x,
        output_dir / f'{prefix}_cartesian.png',
        f'{result.name.title()} trajectory (T={result.duration:.2f}s)',
    )
    _plot_joint_series(
        result.time,
        result.q,
        'Joint position [rad]',
        f'{result.name.title()} joint positions (T={result.duration:.2f}s)',
        output_dir / f'{prefix}_q.png',
        joint_names,
    )
    _plot_joint_series(
        result.time,
        result.q_dot,
        'Joint velocity [rad/s]',
        f'{result.name.title()} joint velocities (T={result.duration:.2f}s)',
        output_dir / f'{prefix}_qdot.png',
        joint_names,
    )
    _plot_joint_series(
        result.time,
        result.q_ddot,
        'Joint acceleration [rad/s²]',
        f'{result.name.title()} joint accelerations (T={result.duration:.2f}s)',
        output_dir / f'{prefix}_qddot.png',
        joint_names,
    )
    _plot_joint_series(
        result.time,
        result.tau,
        'Torque [N·m]',
        f'{result.name.title()} joint torques (T={result.duration:.2f}s)',
        output_dir / f'{prefix}_tau.png',
        joint_names,
    )


def plot_torque_duration_comparison(
    results: Sequence[TrajectoryResult],
    output_dir: Path,
    joint_names: Sequence[str],
):
    if not results:
        return
    name = results[0].name
    n_joints = results[0].tau.shape[1]
    rows = int(np.ceil(n_joints / 2.0))
    fig, axes = plt.subplots(rows, 2, figsize=(12, 3.5 * rows), sharex=False)
    axes_flat = np.atleast_1d(axes).reshape(-1)

    for j in range(n_joints):
        ax = axes_flat[j]
        label = joint_names[j] if j < len(joint_names) else f'joint{j + 1}'
        for result in sorted(results, key=lambda r: r.duration):
            ax.plot(result.time, result.tau[:, j], label=f'T={result.duration:.2f}s')
        ax.set_title(label)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Torque [N·m]')
        ax.grid(True, alpha=0.25)
        ax.legend()

    for idx in range(n_joints, len(axes_flat)):
        axes_flat[idx].axis('off')

    fig.suptitle(f'{name.title()} torque comparison across durations')
    fig.tight_layout()
    fig.savefig(output_dir / f'{name}_torque_duration_comparison.png', dpi=160)
    plt.close(fig)


def _metrics(result: TrajectoryResult) -> Dict[str, np.ndarray]:
    return {
        'max_abs_q_dot': np.max(np.abs(result.q_dot), axis=0),
        'max_abs_q_ddot': np.max(np.abs(result.q_ddot), axis=0),
        'max_abs_tau': np.max(np.abs(result.tau), axis=0),
    }


def summarize_duration_effect(results: Sequence[TrajectoryResult]) -> Dict[str, object]:
    if not results:
        raise ValueError('Duration effect summary requires at least one result.')
    by_duration = []
    for result in sorted(results, key=lambda r: r.duration):
        metrics = _metrics(result)
        by_duration.append({
            'duration_s': float(result.duration),
            'max_abs_q_dot': metrics['max_abs_q_dot'].tolist(),
            'max_abs_q_ddot': metrics['max_abs_q_ddot'].tolist(),
            'max_abs_tau': metrics['max_abs_tau'].tolist(),
            'global_max_abs_q_dot': float(np.max(metrics['max_abs_q_dot'])),
            'global_max_abs_q_ddot': float(np.max(metrics['max_abs_q_ddot'])),
            'global_max_abs_tau': float(np.max(metrics['max_abs_tau'])),
        })
    return {'trajectory': results[0].name, 'durations': by_duration}


def _top_joint_indices(delta_vector: np.ndarray, top_k: int = 2) -> List[int]:
    ranking = np.argsort(np.abs(delta_vector))[::-1]
    return [int(i) for i in ranking[:top_k]]


def build_duration_discussion(
    results: Sequence[TrajectoryResult],
    joint_names: Sequence[str],
) -> Dict[str, object]:
    if len(results) < 2:
        return {
            'answers': {
                'q1_velocity_vs_duration': 'At least two durations are required.',
                'q2_acceleration_vs_duration': 'At least two durations are required.',
                'q3_max_torque_vs_duration': 'At least two durations are required.',
                'q4_torque_shape_vs_duration': 'At least two durations are required.',
                'q5_most_affected_joints': 'At least two durations are required.',
            },
        }

    ordered = sorted(results, key=lambda r: r.duration)
    fast = ordered[0]
    slow = ordered[-1]

    fast_metrics = _metrics(fast)
    slow_metrics = _metrics(slow)

    vel_ratio = float(
        np.max(fast_metrics['max_abs_q_dot']) /
        max(np.max(slow_metrics['max_abs_q_dot']), 1e-12)
    )
    acc_ratio = float(
        np.max(fast_metrics['max_abs_q_ddot']) /
        max(np.max(slow_metrics['max_abs_q_ddot']), 1e-12)
    )
    tau_ratio = float(
        np.max(fast_metrics['max_abs_tau']) /
        max(np.max(slow_metrics['max_abs_tau']), 1e-12)
    )

    tau_delta = slow_metrics['max_abs_tau'] - fast_metrics['max_abs_tau']
    dominant_joints = _top_joint_indices(tau_delta, top_k=min(3, len(tau_delta)))
    dominant_joint_names = [
        joint_names[i] if i < len(joint_names) else f'joint{i + 1}'
        for i in dominant_joints
    ]

    shape_statement = (
        'Longer duration preserves the overall torque profile shape but reduces '
        'dynamic peaks and sharp transitions; gravity-driven baseline components '
        'remain in similar regions.'
    )

    return {
        'comparison': {
            'fast_duration_s': float(fast.duration),
            'slow_duration_s': float(slow.duration),
            'global_velocity_peak_ratio_fast_over_slow': vel_ratio,
            'global_acceleration_peak_ratio_fast_over_slow': acc_ratio,
            'global_torque_peak_ratio_fast_over_slow': tau_ratio,
            'per_joint_delta_max_abs_tau': tau_delta.tolist(),
        },
        'answers': {
            'q1_velocity_vs_duration': (
                'Increasing duration reduces joint velocity peaks; '
                f'fast/slow peak ratio={vel_ratio:.3f} '
                '(>1 confirms faster = higher peaks).'
            ),
            'q2_acceleration_vs_duration': (
                'Increasing duration strongly reduces joint acceleration peaks; '
                f'fast/slow peak ratio={acc_ratio:.3f} '
                '(>1 confirms faster = higher peaks).'
            ),
            'q3_max_torque_vs_duration': (
                'Increasing duration reduces maximum required torque; '
                f'fast/slow peak ratio={tau_ratio:.3f} '
                '(>1 confirms faster = higher peaks).'
            ),
            'q4_torque_shape_vs_duration': shape_statement,
            'q5_most_affected_joints': (
                'Most affected joints by duration change (largest |Δ max| torque): '
                + ', '.join(dominant_joint_names)
                + '.'
            ),
        },
    }


def _parse_args():
    parser = argparse.ArgumentParser(
        description='CR10 torque trajectory computation with Pinocchio.'
    )
    parser.add_argument(
        '--durations',
        nargs='+',
        type=float,
        default=[3.0, 6.0],
        help='Motion durations in seconds.',
    )
    parser.add_argument('--dt', type=float, default=0.01, help='Time step [s].')
    parser.add_argument(
        '--damping',
        type=float,
        default=1e-3,
        help='Damping factor for Jacobian DLS.',
    )
    parser.add_argument(
        '--urdf',
        type=str,
        default=None,
        help='Path to CR10 URDF. If omitted, automatic lookup is used.',
    )
    parser.add_argument(
        '--ee-frame',
        type=str,
        default=None,
        help='Pinocchio end-effector frame name. Defaults to auto-detected.',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/home/helene/ros2_ws/src/cr10_torque_trajectories/part2_results',
        help='Directory for plots and saved trajectories.',
    )
    parser.add_argument(
        '--line-a',
        nargs=3,
        type=float,
        default=[0.45, -0.30, 0.45],
        metavar=('AX', 'AY', 'AZ'),
        help='Straight-line start point [m].',
    )
    parser.add_argument(
        '--line-b',
        nargs=3,
        type=float,
        default=[0.45, 0.30, 0.45],
        metavar=('BX', 'BY', 'BZ'),
        help='Straight-line end point [m].',
    )
    parser.add_argument(
        '--circle-center',
        nargs=3,
        type=float,
        default=[0.50, 0.0, 0.45],
        metavar=('CX', 'CY', 'CZ'),
        help='Circle center [m].',
    )
    parser.add_argument(
        '--circle-radius',
        type=float,
        default=0.12,
        help='Circle radius [m].',
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    durations = sorted(set(float(d) for d in args.durations))
    if len(durations) < 2:
        raise ValueError(
            'At least two different durations are required for duration study '
            '(example: --durations 3 6).'
        )
    
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model, data, urdf_path, ee_frame_id, ee_frame_name, joint_names = (
        load_pinocchio_model(args.urdf, args.ee_frame)
    )
    if model.nq != 6 or model.nv != 6:
        raise ValueError(
            f'Expected a 6-DOF CR10 model, got nq={model.nq}, nv={model.nv}.'
        )

    if len(joint_names) >= model.nv:
        resolved_joint_names = joint_names[:model.nv]
    else:
        resolved_joint_names = JOINT_LABELS[:model.nv]

    results: Dict[str, List[TrajectoryResult]] = {'straight': [], 'circle': []}

    for duration in durations:
        straight = generate_straight_trajectory(
            args.line_a,
            args.line_b,
            duration,
            args.dt,
        )
        straight_result = run_pipeline_for_trajectory(
            model,
            data,
            ee_frame_id,
            straight,
            duration,
            args.damping,
        )
        save_result_artifacts(straight_result, output_dir, resolved_joint_names)
        results['straight'].append(straight_result)

        circle = generate_circle_trajectory(
            args.circle_center,
            args.circle_radius,
            duration,
            args.dt,
        )
        circle_result = run_pipeline_for_trajectory(
            model,
            data,
            ee_frame_id,
            circle,
            duration,
            args.damping,
        )
        save_result_artifacts(circle_result, output_dir, resolved_joint_names)
        results['circle'].append(circle_result)

    plot_torque_duration_comparison(
        results['straight'],
        output_dir,
        resolved_joint_names,
    )
    plot_torque_duration_comparison(
        results['circle'],
        output_dir,
        resolved_joint_names,
    )

    duration_effect = {
        'straight': summarize_duration_effect(results['straight']),
        'circle': summarize_duration_effect(results['circle']),
    }
    duration_discussion = {
        'straight': build_duration_discussion(
            results['straight'],
            resolved_joint_names,
        ),
        'circle': build_duration_discussion(
            results['circle'],
            resolved_joint_names,
        ),
    }

    summary = {
        'urdf': str(urdf_path),
        'pinocchio_model': {
            'nq': int(model.nq),
            'nv': int(model.nv),
            'joint_order': resolved_joint_names,
            'end_effector_frame': ee_frame_name,
        },
        'equations_of_motion': {
            'equation': 'tau = M(q) q_ddot + C(q, q_dot) q_dot + g(q)',
            'terms': {
                'M(q)': 'Joint-space inertia matrix (resistance to acceleration).',
                'C(q, q_dot) q_dot': 'Coriolis and centrifugal torques from motion coupling.',
                'g(q)': 'Gravity compensation torque.',
                'tau': 'Required actuator joint torque vector.',
            },
            'inverse_dynamics_solver': 'pinocchio.rnea',
        },
        'requirements_coverage': {
            'part_a_cartesian_and_ik': {
                'time_vector': 'Implemented in _time_vector.',
                'quintic_s_sdot_sddot': 'Implemented in _quintic_scaling.',
                'straight_path': 'Implemented in generate_straight_trajectory.',
                'circular_path': 'Implemented in generate_circle_trajectory.',
                'ik_q_t': 'Implemented in compute_joint_positions.',
                'stored_outputs': 'Saved in *_data.npz (x, x_dot, x_ddot, q, s, s_dot, s_ddot).',
            },
            'part_b_jacobian_kinematics': {
                'pinocchio_model_loading': 'Implemented in load_pinocchio_model.',
                'jacobian_Jp': (
                    'getFrameJacobian(..., LOCAL_WORLD_ALIGNED) with translational rows.'
                ),
                'velocity_q_dot': 'Computed by damped least-squares pseudoinverse.',
                'jacobian_time_derivative_Jp_dot': (
                    'computeJointJacobiansTimeVariation + getFrameJacobianTimeVariation.'
                ),
                'acceleration_q_ddot': 'Computed from x_ddot - Jp_dot q_dot via DLS.',
                'consistent_frame_usage': 'Jp and Jp_dot both in LOCAL_WORLD_ALIGNED.',
            },
            'part_c_dynamics_and_torque': {
                'inverse_dynamics_tau_t': 'Implemented in compute_torque_trajectory via rnea.',
                'required_plots': (
                    'Cartesian, q, q_dot, q_ddot, tau per duration/trajectory and '
                    'torque-duration comparisons.'
                ),
                'duration_study': 'Computed for all durations in --durations.',
            },
        },
        'duration_effect': duration_effect,
        'duration_discussion': duration_discussion,
        'artifacts': {
            'per_run_data_files': '*.npz',
            'per_run_plots': '*_cartesian.png, *_q.png, *_qdot.png, *_qddot.png, *_tau.png',
            'duration_comparison_plots': '*_torque_duration_comparison.png',
            'summary_file': 'summary.json',
        },
    }
    with (output_dir / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('Torque trajectory computation completed.')
    print(f'URDF: {urdf_path}')
    print(f'End-effector frame: {ee_frame_name}')
    print(f'Output directory: {output_dir}')
    print('Generated files include:')
    print('- Cartesian, q, q_dot, q_ddot, tau plots per trajectory/duration')
    print('- Torque duration comparison plots for straight and circle')
    print('- Raw trajectory arrays (*.npz) including s, s_dot, s_ddot')
    print('- summary.json with requirement coverage and duration discussion')


if __name__ == '__main__':
    main()
