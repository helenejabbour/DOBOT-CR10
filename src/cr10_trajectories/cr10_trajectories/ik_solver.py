#!/usr/bin/python3
"""
ik_solver.py

Inverse kinematics solver and ROS integration layer for Dobot CR10.

This module combines:
  - FK and Jacobian from fk_model.py (symbolic + compiled to numeric)
  - DLS solver from ik_model.py (iterative IK with adaptive damping)
  - Candidate branch selection (exhaustive multi-seed search vs. fast local tracking)
  - Manipulability-aware IK scoring (Yoshikawa position manipulability metric)

Public API:
  - solve_ik_candidates(...): Multi-seed branch search for branch selection or fallback
  - solve_ik_direct(...): Fast single-shot warm-started solve for dense trajectory tracking
  - solve_ik(...): Legacy single-solution wrapper
  - forward_kinematics(...): FK evaluation
  - is_reachable(...): Workspace checking

CR10 Workspace:
  - Max reach: 1.3 m from base
  - Min reach: 0.2 m from base
  - IK target tolerance: 5 mm position, 6° orientation
"""
import numpy as np

from .fk_model import get_ee_position, get_fk, get_jacobian
from .ik_model import solve_ik_dls

CR10_MAX_REACH = 1.300
CR10_MIN_REACH = 0.2
IK_POSITION_TOL = 0.005
DOWN_AXIS = np.array([0.0, 0.0, -1.0], dtype=float)
DEFAULT_SEED = np.array([0.0, -0.5, 0.5, 0.0, 0.0, 0.0], dtype=float)
MANIPULABILITY_EPS = 1e-4

_prev_theta = DEFAULT_SEED.copy()


def reset_ik_seed():
    global _prev_theta
    _prev_theta = DEFAULT_SEED.copy()


def set_ik_seed(theta):
    global _prev_theta
    _prev_theta = np.array(theta, dtype=float).copy()


def is_reachable(p):
    r = np.linalg.norm(p)
    return CR10_MIN_REACH < r < CR10_MAX_REACH


def _tool_down_angle_deg(theta):
    """Angle between current tool Z-axis and world down direction."""
    T = np.array(get_fk(theta), dtype=float)
    tool_z = T[:3, 2]
    cos_angle = np.clip(np.dot(tool_z, DOWN_AXIS) / np.linalg.norm(tool_z), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def _build_tool_down_rotation(target):
    """Build a practical orientation target with tool Z-axis pointing downward."""
    z_axis = DOWN_AXIS
    radial = np.array([target[0], target[1], 0.0], dtype=float)
    if np.linalg.norm(radial) < 1e-6:
        radial = np.array([1.0, 0.0, 0.0], dtype=float)
    x_axis = radial / np.linalg.norm(radial)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def _position_manipulability(theta):
    """Yoshikawa manipulability from position Jacobian only."""
    J = np.array(get_jacobian(theta), dtype=float)
    Jp = J[:3, :]
    JJt = Jp @ Jp.T
    det_val = float(np.linalg.det(JJt))
    return np.sqrt(max(det_val, 0.0))


def _wrap_to_pi(theta):
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def _elbow_sign(theta):
    return 1 if float(theta[2]) >= 0.0 else -1


def _candidate_seeds(seed, target, exhaustive=True):
    """Generate shoulder/elbow/wrist branch seeds for one target."""
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
        # Fast local search around the current branch for dense trajectories.
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
    x,
    y,
    z,
    reference_theta=None,
    elbow_mode='auto',
    max_solutions=8,
    exhaustive=True,
):
    """
    Find multiple IK solutions ranked by continuity, tool orientation, and manipulability.
    
    Args:
        x, y, z: Target Cartesian position
        reference_theta: Seed joint configuration (default: last known pose)
        elbow_mode: 'auto', 'up', or 'down' to filter by elbow sign
        max_solutions: Maximum candidates to return (sorted by score)
        exhaustive: If True, try many seed branches; if False, local search only
    
    Returns:
        List of dicts with keys: 'angles', 'position_error', 'tool_down_angle_deg', 'manipulability', 'score'
    
    Scoring favors:
        - Low tool angle deviation from downward (tool-down approach)
        - High manipulability (avoid singularities)
        - Continuity to reference pose
    """
    target = np.array([x, y, z], dtype=float)
    reference = _prev_theta if reference_theta is None else np.array(reference_theta, dtype=float)
    target_rot = _build_tool_down_rotation(target)

    if exhaustive:
        attempts = [
            {'orientation_weight': 0.18, 'rot_tolerance': np.deg2rad(8.0), 'max_iterations': 550},
            {'orientation_weight': 0.12, 'rot_tolerance': np.deg2rad(12.0), 'max_iterations': 500},
            {'orientation_weight': 0.06, 'rot_tolerance': np.deg2rad(18.0), 'max_iterations': 450},
        ]
    else:
        attempts = [
            {'orientation_weight': 0.12, 'rot_tolerance': np.deg2rad(12.0), 'max_iterations': 280},
            {'orientation_weight': 0.06, 'rot_tolerance': np.deg2rad(18.0), 'max_iterations': 240},
        ]

    solutions = []
    solve_errors = []
    for seed in _candidate_seeds(reference, target, exhaustive=exhaustive):
        for cfg in attempts:
            try:
                theta_candidate = solve_ik_dls(
                    target,
                    target_rot=target_rot,
                    initial_theta=seed,
                    orientation_weight=cfg['orientation_weight'],
                    max_iterations=cfg['max_iterations'],
                    pos_tolerance=1e-3,
                    rot_tolerance=cfg['rot_tolerance'],
                )
            except ValueError as exc:
                solve_errors.append(str(exc))
                continue

            ee_position = np.array(get_ee_position(theta_candidate), dtype=float)
            position_error = np.linalg.norm(target - ee_position)
            if position_error > IK_POSITION_TOL:
                continue

            elbow_sign = _elbow_sign(theta_candidate)
            if elbow_mode == 'up' and elbow_sign != -1:
                continue
            if elbow_mode == 'down' and elbow_sign != 1:
                continue

            tool_down_angle = _tool_down_angle_deg(theta_candidate)
            continuity_cost = np.linalg.norm(_wrap_to_pi(theta_candidate - reference))
            manipulability = _position_manipulability(theta_candidate)
            singularity_cost = 1.0 / (manipulability + MANIPULABILITY_EPS)
            score = 5.0 * tool_down_angle + 0.5 * continuity_cost + 0.05 * singularity_cost

            duplicate = False
            for prev in solutions:
                diff = np.linalg.norm(_wrap_to_pi(theta_candidate - prev['angles']))
                if diff < 0.08:
                    duplicate = True
                    if score < prev['score']:
                        prev.update({
                            'angles': theta_candidate.copy(),
                            'position_error': float(position_error),
                            'tool_down_angle_deg': float(tool_down_angle),
                            'elbow_sign': elbow_sign,
                            'manipulability': float(manipulability),
                            'score': float(score),
                        })
                    break
            if duplicate:
                continue

            solutions.append({
                'angles': theta_candidate.copy(),
                'position_error': float(position_error),
                'tool_down_angle_deg': float(tool_down_angle),
                'elbow_sign': elbow_sign,
                'manipulability': float(manipulability),
                'score': float(score),
            })

    if not solutions:
        raise ValueError(
            'IK failed to find valid candidates. '
            f'Details: {" | ".join(solve_errors[:4])}'
        )

    solutions.sort(key=lambda item: item['score'])
    return solutions[:max_solutions]


def solve_ik_direct(
    x,
    y,
    z,
    reference_theta=None,
    orientation_weight=0.10,
    max_iterations=180,
):
    """
    Fast single-shot IK solve for waypoint-to-waypoint trajectory tracking.
    
    This method uses warm-start DLS from a reference pose, skipping branch exploration.
    Ideal for dense trajectories where continuity from the previous point is guaranteed.
    
    Args:
        x, y, z: Target Cartesian position
        reference_theta: Warm-start joint configuration (required; no branch search)
        orientation_weight: Cost weight for orientation error (0-1)
        max_iterations: Max DLS iterations (default 180 for speed)
    
    Returns:
        Joint angles as ndarray
    
    Raises:
        ValueError: If final position error exceeds 5 mm tolerance
    """
    target = np.array([x, y, z], dtype=float)
    reference = _prev_theta if reference_theta is None else np.array(reference_theta, dtype=float)
    target_rot = _build_tool_down_rotation(target)

    theta = solve_ik_dls(
        target,
        target_rot=target_rot,
        initial_theta=reference,
        orientation_weight=orientation_weight,
        max_iterations=max_iterations,
        pos_tolerance=1e-3,
        rot_tolerance=np.deg2rad(18.0),
    )

    ee_position = np.array(get_ee_position(theta), dtype=float)
    position_error = np.linalg.norm(target - ee_position)
    if position_error > IK_POSITION_TOL:
        raise ValueError(
            f'Fast IK validation failed: |target - fk(q)|={position_error:.4f}m '
            f'(tol={IK_POSITION_TOL:.4f}m).'
        )

    return np.array(theta, dtype=float)


def solve_ik(x, y, z, elbow_mode='auto'):
    global _prev_theta
    candidates = solve_ik_candidates(
        x,
        y,
        z,
        reference_theta=_prev_theta,
        elbow_mode=elbow_mode,
        max_solutions=1,
    )
    theta = np.array(candidates[0]['angles'], dtype=float)

    ee_position = np.array(get_ee_position(theta), dtype=float)
    target = np.array([x, y, z], dtype=float)
    position_error = np.linalg.norm(target - ee_position)
    if position_error > IK_POSITION_TOL:
        raise ValueError(
            f'IK validation failed: |target - fk(q)|={position_error:.4f}m '
            f'(tol={IK_POSITION_TOL:.4f}m).'
        )

    _prev_theta = theta.copy()
    return list(theta)


def forward_kinematics(joint_angles):
    return get_ee_position(joint_angles)
