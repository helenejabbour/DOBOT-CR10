#!/usr/bin/python3
"""Inverse kinematics solver utilities for Dobot CR10."""
import numpy as np

from .ik_model import solve_ik_dls

try:
    import pinocchio as pin
except ImportError as import_error:  # pragma: no cover
    pin = None
    _PINOCCHIO_IMPORT_ERROR = import_error
else:
    _PINOCCHIO_IMPORT_ERROR = None

CR10_MAX_REACH = 1.300
CR10_MIN_REACH = 0.2
IK_POSITION_TOL = 0.005
DEFAULT_SEED = np.array([0.0, -0.5, 0.5, 0.0, 0.0, 0.0], dtype=float)
MANIPULABILITY_EPS = 1e-4

_prev_theta = DEFAULT_SEED.copy()


def _require_pinocchio():
    if pin is None:  # pragma: no cover
        raise ImportError(
            'pinocchio is required for IK.'
        ) from _PINOCCHIO_IMPORT_ERROR


def reset_ik_seed():
    global _prev_theta
    _prev_theta = DEFAULT_SEED.copy()


def set_ik_seed(theta):
    global _prev_theta
    _prev_theta = np.array(theta, dtype=float).copy()


def is_reachable(p):
    r = np.linalg.norm(p)
    return CR10_MIN_REACH < r < CR10_MAX_REACH


def forward_kinematics(model, data, frame_id, joint_angles):
    _require_pinocchio()
    q = np.asarray(joint_angles, dtype=float)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return np.array(data.oMf[frame_id].translation, dtype=float)


def _position_manipulability(model, data, frame_id, theta):
    _require_pinocchio()
    q = np.asarray(theta, dtype=float)
    pin.forwardKinematics(model, data, q)
    pin.computeJointJacobians(model, data, q)
    pin.updateFramePlacements(model, data)
    J = pin.getFrameJacobian(
        model,
        data,
        frame_id,
        pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
    )
    Jp = np.asarray(J[:3, :], dtype=float)
    JJt = Jp @ Jp.T
    det_val = float(np.linalg.det(JJt))
    return np.sqrt(max(det_val, 0.0))


def _wrap_to_pi(theta):
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def _elbow_sign(theta):
    return 1 if float(theta[2]) >= 0.0 else -1


def _candidate_seeds(seed, target, exhaustive=True):
    variants = []
    q1_guess = np.arctan2(float(target[1]), float(target[0]))

    base = np.array(seed, dtype=float)
    elbow_flip = base.copy()
    elbow_flip[1] = -elbow_flip[1]
    elbow_flip[2] = -elbow_flip[2]

    canonical = [
        np.array([q1_guess, -1.00, 1.20, 0.0, 0.0, 0.0], dtype=float),
        np.array([q1_guess, 1.00, -1.20, 0.0, 0.0, 0.0], dtype=float),
        np.array([q1_guess, -1.35, 1.55, 0.0, 0.0, 0.0], dtype=float),
        np.array([q1_guess, 1.35, -1.55, 0.0, 0.0, 0.0], dtype=float),
    ]

    if exhaustive:
        base_variants = [base, elbow_flip] + canonical
        wrist_offsets = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
            np.array([0.0, 0.0, 0.0, np.pi, 0.0, 0.0], dtype=float),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.pi], dtype=float),
            np.array([0.0, 0.0, 0.0, np.pi, 0.0, np.pi], dtype=float),
        ]
    else:
        base_variants = [base, elbow_flip, canonical[0], canonical[1]]
        wrist_offsets = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
            np.array([0.0, 0.0, 0.0, np.pi, 0.0, 0.0], dtype=float),
        ]

    seen = set()
    for base_seed in base_variants:
        for wrist in wrist_offsets:
            q = _wrap_to_pi(base_seed + wrist)
            sig = tuple(np.round(q, 3))
            if sig in seen:
                continue
            seen.add(sig)
            variants.append(q)

    return variants


def solve_ik_candidates(
    model,
    data,
    frame_id,
    x,
    y,
    z,
    reference_theta=None,
    elbow_mode='auto',
    max_solutions=8,
    exhaustive=True,
    q6_ref=None,
):
    _require_pinocchio()
    target = np.array([x, y, z], dtype=float)
    reference = (
        _prev_theta if reference_theta is None
        else np.array(reference_theta, dtype=float)
    )

    if exhaustive:
        attempts = [
            {
                'nullspace_weight': 0.8,
                'damping': 0.01,
                'max_iterations': 550,
            },
            {
                'nullspace_weight': 0.5,
                'damping': 0.01,
                'max_iterations': 500,
            },
            {
                'nullspace_weight': 0.2,
                'damping': 0.01,
                'max_iterations': 450,
            },
        ]
    else:
        attempts = [
            {
                'nullspace_weight': 0.8,
                'damping': 0.01,
                'max_iterations': 280,
            },
            {
                'nullspace_weight': 0.5,
                'damping': 0.01,
                'max_iterations': 240,
            },
        ]

    solutions = []
    solve_errors = []
    for seed in _candidate_seeds(reference, target, exhaustive=exhaustive):
        for cfg in attempts:
            try:
                theta_candidate = solve_ik_dls(
                    model,
                    data,
                    frame_id,
                    target,
                    initial_theta=seed,
                    q_ref=reference,
                    nullspace_weight=cfg['nullspace_weight'],
                    damping=cfg['damping'],
                    center_weight=0.05,
                    q6_ref=q6_ref,
                    q6_weight=0.15,
                    max_iterations=cfg['max_iterations'],
                    pos_tolerance=1e-3,
                )
            except ValueError as exc:
                solve_errors.append(str(exc))
                continue

            ee_position = forward_kinematics(model, data, frame_id, theta_candidate)
            position_error = np.linalg.norm(target - ee_position)
            if position_error > IK_POSITION_TOL:
                continue

            elbow_sign = _elbow_sign(theta_candidate)
            if elbow_mode == 'up' and elbow_sign != -1:
                continue
            if elbow_mode == 'down' and elbow_sign != 1:
                continue

            continuity_delta = _wrap_to_pi(theta_candidate - reference)
            continuity_weights = np.ones_like(continuity_delta)
            if continuity_weights.size >= 6:
                continuity_weights[5] = 0.25
            continuity_cost = np.linalg.norm(
                continuity_weights * continuity_delta
            )
            q6_cost = 0.0
            if q6_ref is not None and theta_candidate.size >= 6:
                q6_cost = abs(float(theta_candidate[5] - q6_ref))
            manipulability = _position_manipulability(
                model,
                data,
                frame_id,
                theta_candidate,
            )
            singularity_cost = 1.0 / (manipulability + MANIPULABILITY_EPS)
            score = 2.0 * continuity_cost + 0.1 * singularity_cost + 0.5 * q6_cost

            solutions.append({
                'angles': theta_candidate.copy(),
                'position_error': float(position_error),
                'elbow_sign': elbow_sign,
                'manipulability': float(manipulability),
                'score': float(score),
            })

    if not solutions:
        details = ' | '.join(solve_errors[:4])
        raise ValueError(f'IK failed to find valid candidates. {details}')

    solutions.sort(key=lambda item: item['score'])
    return solutions[:max_solutions]


def solve_ik_direct(
    model,
    data,
    frame_id,
    x,
    y,
    z,
    reference_theta=None,
    max_iterations=180,
    q6_ref=None,
):
    _require_pinocchio()
    target = np.array([x, y, z], dtype=float)
    reference = (
        _prev_theta if reference_theta is None
        else np.array(reference_theta, dtype=float)
    )

    theta = solve_ik_dls(
        model,
        data,
        frame_id,
        target,
        initial_theta=reference,
        q_ref=reference,
        nullspace_weight=0.8,
        damping=0.01,
        center_weight=0.05,
        q6_ref=q6_ref,
        q6_weight=0.15,
        max_iterations=max_iterations,
        pos_tolerance=1e-3,
    )

    ee_position = forward_kinematics(model, data, frame_id, theta)
    position_error = np.linalg.norm(target - ee_position)
    if position_error > IK_POSITION_TOL:
        raise ValueError(
            f'Fast IK validation failed: error={position_error:.4f}m '
            f'(tol={IK_POSITION_TOL:.4f}m).'
        )

    return np.array(theta, dtype=float)
