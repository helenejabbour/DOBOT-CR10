#!/usr/bin/python3
"""Damped Least Squares inverse kinematics solver for Dobot CR10."""

import numpy as np

try:
	import pinocchio as pin
except ImportError as import_error:  # pragma: no cover
	pin = None
	_PINOCCHIO_IMPORT_ERROR = import_error
else:
	_PINOCCHIO_IMPORT_ERROR = None

max_iter = 500
pos_tol = 1e-3

JOINT_MIN = np.array([-6.2832, -6.2832, -2.7925, -6.2832, -6.2832, -6.2832])
JOINT_MAX = np.array([6.2832, 6.2832, 2.7925, 6.2832, 6.2832, 6.2832])


def solve_ik_dls(
	model,
	data,
	frame_id,
	target_pos,
	initial_theta=None,
	q_ref=None,
	nullspace_weight=0.5,
	damping=0.01,
	center_weight=0.05,
	q6_ref=None,
	q6_weight=0.15,
	max_iterations=None,
	pos_tolerance=None,
):
	if pin is None:  # pragma: no cover
		raise ImportError(
			'pinocchio is required for IK.'
		) from _PINOCCHIO_IMPORT_ERROR

	if initial_theta is None:
		initial_theta = np.zeros(model.nv)
	if max_iterations is None:
		max_iterations = max_iter
	if pos_tolerance is None:
		pos_tolerance = pos_tol

	theta = np.array(initial_theta, dtype=float).copy()
	target_pos = np.asarray(target_pos, dtype=float)
	if q_ref is not None:
		q_ref = np.asarray(q_ref, dtype=float)
	nullspace_weight = float(nullspace_weight)
	damping = float(max(damping, 1e-8))
	center_weight = float(max(center_weight, 0.0))
	q6_weight = float(max(q6_weight, 0.0))
	joint_center = 0.5 * (JOINT_MIN[:model.nv] + JOINT_MAX[:model.nv])

	for _ in range(max_iterations):
		pin.forwardKinematics(model, data, theta)
		pin.computeJointJacobians(model, data, theta)
		pin.updateFramePlacements(model, data)

		current_pos = np.asarray(data.oMf[frame_id].translation, dtype=float)
		pos_error = target_pos - current_pos

		if np.linalg.norm(pos_error) < pos_tolerance:
			return theta

		J_full = pin.getFrameJacobian(
			model,
			data,
			frame_id,
			pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
		)
		Jp = np.asarray(J_full[:3, :], dtype=float)
		if (
			np.linalg.matrix_rank(Jp) < 3 and
			np.linalg.norm(pos_error) > 5.0 * pos_tolerance
		):
			raise ValueError(
				'Singularity detected: position Jacobian rank dropped below 3.'
			)

		lam2_I = (damping ** 2) * np.eye(3, dtype=float)
		Jp_pinv = Jp.T @ np.linalg.solve(Jp @ Jp.T + lam2_I, np.eye(3))
		dtheta_primary = Jp_pinv @ pos_error
		N = np.eye(model.nv, dtype=float) - Jp_pinv @ Jp

		secondary_velocity = np.zeros(model.nv, dtype=float)
		if q_ref is not None:
			secondary_velocity += -nullspace_weight * (theta - q_ref)
		if center_weight > 0.0:
			secondary_velocity += center_weight * (joint_center - theta)
		if q6_ref is not None and model.nv >= 6 and q6_weight > 0.0:
			# Linear (non-wrapped) attraction prevents 2π-equivalent winding drift.
			secondary_velocity[5] += q6_weight * (float(q6_ref) - theta[5])

		dtheta_secondary = N @ secondary_velocity

		dtheta = dtheta_primary + dtheta_secondary
		alpha = min(0.25, 1.0 / (1.0 + np.linalg.norm(dtheta)))
		theta += alpha * dtheta
		theta = np.clip(theta, JOINT_MIN[:model.nv], JOINT_MAX[:model.nv])

	raise ValueError(
		f'IK did not converge after {max_iterations} iterations. '
		f'Target={target_pos}.'
	)
