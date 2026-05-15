#!/usr/bin/python3
"""
trajectory_node.py

Core ROS 2 node for Dobot CR10 inverse kinematics and trajectory control.

This node:
  - Receives Cartesian trajectory commands (straight-line or circular)
  - Plans joint-space trajectories using sequential warm-started IK and fast local tracking
  - Publishes joint states to control the robot in RViz
  - Visualizes the end-effector trace path and waypoint markers
  - Uses time-parameterized motion with joint velocity and acceleration limits
  - Implements manipulability-aware IK candidate scoring to avoid singularities

Trajectory Planning Strategy:
  - Dense paths (>= 40 points): Fast sequential IK with warm-starting and fallback candidate search
  - Sparse paths: Full candidate branch search with dynamic programming path optimization
  - All paths: Adaptive waypoint density based on geometry and requested speed

Subscriptions:
  - target_points (std_msgs/Float64MultiArray): [Ax, Ay, Az, Bx, By, Bz, speed, elbow_mode?]
  - circle_params (std_msgs/Float64MultiArray): [Cx, Cy, Cz, radius, speed, elbow_mode?]

Publications:
  - /joint_states (sensor_msgs/JointState): Current joint configuration
  - /ee_trace (visualization_msgs/Marker): End-effector path as line strip
  - /ee_trace_array (visualization_msgs/MarkerArray): Waypoint spheres
  - trajectory_status (std_msgs/Float64MultiArray): Status code for GUI feedback
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import Float64MultiArray
import numpy as np
import time

from .ik_solver import (
    forward_kinematics,
    is_reachable,
    reset_ik_seed,
    set_ik_seed,
    solve_ik_direct,
    solve_ik_candidates,
)

JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
MIN_Z = 0.15  # minimum Z height to avoid floor collision
EE_TRACKING_TOL = 0.01
MAX_IK_CANDIDATES = 6
FAST_IK_CANDIDATES = 3
MOTION_COST_WEIGHT = 1.0
TOOL_DOWN_COST_WEIGHT = 0.01
MANIPULABILITY_COST_WEIGHT = 0.03
BRANCH_SWITCH_PENALTY = 0.30
MAX_TRAJECTORY_POINTS = 120
PLANNING_TIMEOUT_SEC = 35.0
FAST_TRACK_THRESHOLD = 40
JOINT_VEL_LIMIT_RAD_S = np.array([1.1, 1.1, 1.0, 1.6, 1.6, 2.0], dtype=float)
JOINT_ACC_LIMIT_RAD_S2 = np.array([2.2, 2.2, 2.0, 3.2, 3.2, 4.0], dtype=float)
MIN_SEGMENT_DT = 0.03
MIN_CIRCLE_POINTS = 24


class TrajectoryNode(Node):
    def __init__(self):
        super().__init__('trajectory_node')

        self.js_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.marker_pub = self.create_publisher(Marker, '/ee_trace', 10)
        self.marker_array_pub = self.create_publisher(MarkerArray, '/ee_trace_array', 10)
        self.status_pub = self.create_publisher(Float64MultiArray, 'trajectory_status', 10)

        self.create_subscription(Float64MultiArray, 'target_points', self.handle_straight, 10)
        self.create_subscription(Float64MultiArray, 'circle_params', self.handle_circle, 10)

        self.trace_points = []
        self.waypoint_id = 0
        self.waypoints = []      # list of (angles, cartesian_point)
        self.current_step = 0
        self.motion_active = False
        self.motion_start_time = 0.0
        self.waypoint_times = np.array([], dtype=float)
        self.total_motion_time = 0.0
        self.last_dot_waypoint = -1
        self.last_sample_point = None
        self.cartesian_trajectory = []
        self.control_period = 0.02
        
        self.loading = False
        self.timer = self.create_timer(self.control_period, self.publish_step)
        self.get_logger().info('TrajectoryNode ready. Waiting for commands.')

    # ------------------------------------------------------------------ #
    #  Status publisher → GUI reads this
    # ------------------------------------------------------------------ #
    def send_status(self, code, message):
        """
        code: 0=info, 1=warning, 2=error, 3=success
        Publishes [code] and logs message.
        GUI subscribes and displays it.
        """
        msg = Float64MultiArray()
        msg.data = [float(code)]
        self.status_pub.publish(msg)
        if code == 2:
            self.get_logger().error(message)
        elif code == 1:
            self.get_logger().warn(message)
        else:
            self.get_logger().info(message)

    # ------------------------------------------------------------------ #
    #  Clear markers
    # ------------------------------------------------------------------ #
    def clear_markers(self):
        m = Marker()
        m.header.frame_id = 'base_link'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'ee_trace'
        m.id = 0
        m.action = Marker.DELETEALL
        self.marker_pub.publish(m)

        ma = MarkerArray()
        del_m = Marker()
        del_m.header.frame_id = 'base_link'
        del_m.header.stamp = self.get_clock().now().to_msg()
        del_m.ns = 'waypoints'
        del_m.action = Marker.DELETEALL
        ma.markers.append(del_m)
        self.marker_array_pub.publish(ma)

        self.trace_points.clear()
        self.waypoint_id = 0

    def _decode_elbow_mode(self, mode_value):
        if mode_value >= 0.5:
            return 'up'
        if mode_value <= -0.5:
            return 'down'
        return 'auto'

    def _joint_delta_norm(self, q_next, q_prev):
        delta = np.array(q_next, dtype=float) - np.array(q_prev, dtype=float)
        delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
        return np.linalg.norm(delta)

    def _shortest_joint_delta(self, q_from, q_to):
        delta = np.array(q_to, dtype=float) - np.array(q_from, dtype=float)
        return (delta + np.pi) % (2.0 * np.pi) - np.pi

    def _segment_duration(self, q_from, q_to):
        dq = np.abs(self._shortest_joint_delta(q_from, q_to))
        t_vel = np.max(dq / np.maximum(JOINT_VEL_LIMIT_RAD_S, 1e-6))
        t_acc = np.max(2.0 * np.sqrt(dq / np.maximum(JOINT_ACC_LIMIT_RAD_S2, 1e-6)))
        return float(max(MIN_SEGMENT_DT, t_vel, t_acc))

    def _spacing_from_speed(self, speed):
        s = float(np.clip(speed, 1.0, 10.0))
        # Lower speed -> denser sampling, higher speed -> lighter sampling.
        return float(np.interp(s, [1.0, 10.0], [0.006, 0.020]))

    def _adaptive_steps_for_line(self, A, B, speed):
        distance = float(np.linalg.norm(np.array(B, dtype=float) - np.array(A, dtype=float)))
        spacing = self._spacing_from_speed(speed)
        raw_steps = int(np.ceil(distance / max(spacing, 1e-4)))
        return int(np.clip(max(10, raw_steps), 10, MAX_TRAJECTORY_POINTS))

    def _adaptive_steps_for_circle(self, radius, speed):
        circumference = float(2.0 * np.pi * radius)
        spacing = self._spacing_from_speed(speed)
        raw_steps = int(np.ceil(circumference / max(spacing, 1e-4)))
        min_steps = max(10, MIN_CIRCLE_POINTS)
        return int(np.clip(max(min_steps, raw_steps), min_steps, MAX_TRAJECTORY_POINTS))

    def _quintic_scale(self, u):
        """Fifth-order time-scaling profile s(u), u in [0, 1]."""
        u = float(np.clip(u, 0.0, 1.0))
        return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5

    def _quintic_scale_dot(self, u, duration):
        u = float(np.clip(u, 0.0, 1.0))
        duration = float(max(duration, 1e-6))
        return (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / duration

    def _quintic_scale_ddot(self, u, duration):
        u = float(np.clip(u, 0.0, 1.0))
        duration = float(max(duration, 1e-6))
        return (60.0 * u - 180.0 * u**2 + 120.0 * u**3) / (duration**2)

    def _sample_time_vector(self, steps, duration):
        duration = float(max(duration, self.control_period))
        return np.linspace(0.0, duration, int(steps) + 1, dtype=float)

    def _sample_straight_trajectory(self, A, B, steps, duration):
        time_vector = self._sample_time_vector(steps, duration)
        A_np = np.array(A, dtype=float)
        delta = np.array(B, dtype=float) - A_np

        positions = []
        velocities = []
        accelerations = []
        for t in time_vector:
            u = t / float(max(duration, 1e-6))
            s = self._quintic_scale(u)
            s_dot = self._quintic_scale_dot(u, duration)
            s_ddot = self._quintic_scale_ddot(u, duration)
            positions.append(A_np + s * delta)
            velocities.append(s_dot * delta)
            accelerations.append(s_ddot * delta)
        return time_vector, positions, velocities, accelerations

    def _sample_circle_trajectory(self, center, radius, steps, duration):
        time_vector = self._sample_time_vector(steps, duration)
        center = np.array(center, dtype=float)
        cx, cy, cz = center
        positions = []
        velocities = []
        accelerations = []
        for t in time_vector:
            u = t / float(max(duration, 1e-6))
            s = self._quintic_scale(u)
            s_dot = self._quintic_scale_dot(u, duration)
            s_ddot = self._quintic_scale_ddot(u, duration)
            theta = 2.0 * np.pi * s
            theta_dot = 2.0 * np.pi * s_dot
            theta_ddot = 2.0 * np.pi * s_ddot
            positions.append(np.array([
                cx + radius * np.cos(theta),
                cy + radius * np.sin(theta),
                cz,
            ], dtype=float))
            velocities.append(np.array([
                -radius * np.sin(theta) * theta_dot,
                radius * np.cos(theta) * theta_dot,
                0.0,
            ], dtype=float))
            accelerations.append(np.array([
                -radius * np.cos(theta) * theta_dot**2 - radius * np.sin(theta) * theta_ddot,
                -radius * np.sin(theta) * theta_dot**2 + radius * np.cos(theta) * theta_ddot,
                0.0,
            ], dtype=float))
        return time_vector, positions, velocities, accelerations

    def _prepare_motion_profile(self):
        if len(self.waypoints) < 2:
            self.waypoint_times = np.array([0.0], dtype=float)
            self.total_motion_time = 0.0
            return

        segment_durations = []
        for i in range(len(self.waypoints) - 1):
            q0 = self.waypoints[i][0]
            q1 = self.waypoints[i + 1][0]
            segment_durations.append(self._segment_duration(q0, q1))

        self.waypoint_times = np.zeros(len(self.waypoints), dtype=float)
        for i, dt in enumerate(segment_durations, start=1):
            self.waypoint_times[i] = self.waypoint_times[i - 1] + dt

        self.total_motion_time = float(self.waypoint_times[-1])

    def _build_optimal_joint_path(self, cartesian_points, elbow_mode='auto'):
        """
        Plan a joint-space trajectory through Cartesian waypoints.
        
        Args:
            cartesian_points: List of [x, y, z] target positions
            elbow_mode: 'auto', 'up' (elbow up), or 'down' (elbow down)
        
        Returns:
            List of (joint_angles, ee_position) tuples for smooth motion
        
        Algorithm:
            - Dense paths (>= FAST_TRACK_THRESHOLD): Fast sequential tracking
            - Sparse paths: Full candidate branch search with DP optimization
        """
        if len(cartesian_points) >= FAST_TRACK_THRESHOLD:
            return self._build_fast_joint_path(cartesian_points, elbow_mode=elbow_mode)

        candidate_sets = []
        reference_theta = None
        plan_start = time.perf_counter()
        waypoint_count = len(cartesian_points)

        for i, p in enumerate(cartesian_points):
            if (time.perf_counter() - plan_start) > PLANNING_TIMEOUT_SEC:
                raise ValueError(
                    f'Planning timeout after {PLANNING_TIMEOUT_SEC:.0f}s at waypoint {i}/{waypoint_count}. '
                    f'Reduce speed detail (fewer steps) or choose a simpler path.'
                )

            exhaustive_search = (i == 0)
            max_solutions = MAX_IK_CANDIDATES if exhaustive_search else FAST_IK_CANDIDATES
            try:
                candidates = solve_ik_candidates(
                    float(p[0]),
                    float(p[1]),
                    float(p[2]),
                    reference_theta=reference_theta,
                    elbow_mode=elbow_mode,
                    max_solutions=max_solutions,
                    exhaustive=exhaustive_search,
                )
            except ValueError as exc:
                raise ValueError(f'IK failed at waypoint {i}: {exc}') from exc

            if not candidates:
                raise ValueError(f'No IK candidates found at waypoint {i}.')

            candidate_sets.append(candidates)

            # Use local continuity to keep planning fast and branch-consistent.
            reference_theta = np.array(candidates[0]['angles'], dtype=float)

            if i > 0 and i % 20 == 0:
                elapsed = time.perf_counter() - plan_start
                self.send_status(
                    0,
                    f'Planning progress: {i}/{waypoint_count} waypoints ({elapsed:.1f}s).',
                )

        prev_costs = []
        backptr = []
        first = candidate_sets[0]
        for cand in first:
            prev_costs.append(
                TOOL_DOWN_COST_WEIGHT * float(cand['tool_down_angle_deg'])
                + MANIPULABILITY_COST_WEIGHT / (float(cand.get('manipulability', 1e-3)) + 1e-3)
            )
        backptr.append([-1] * len(first))

        for i in range(1, len(candidate_sets)):
            prev_candidates = candidate_sets[i - 1]
            curr_candidates = candidate_sets[i]
            curr_costs = [float('inf')] * len(curr_candidates)
            curr_prev = [-1] * len(curr_candidates)

            for j, curr in enumerate(curr_candidates):
                for k, prev in enumerate(prev_candidates):
                    step_cost = MOTION_COST_WEIGHT * self._joint_delta_norm(
                        curr['angles'],
                        prev['angles'],
                    )
                    if curr['elbow_sign'] != prev['elbow_sign']:
                        step_cost += BRANCH_SWITCH_PENALTY
                    total = (
                        prev_costs[k]
                        + step_cost
                        + TOOL_DOWN_COST_WEIGHT * float(curr['tool_down_angle_deg'])
                        + MANIPULABILITY_COST_WEIGHT / (float(curr.get('manipulability', 1e-3)) + 1e-3)
                    )
                    if total < curr_costs[j]:
                        curr_costs[j] = total
                        curr_prev[j] = k

            prev_costs = curr_costs
            backptr.append(curr_prev)

        if not prev_costs:
            return []

        best_last = int(np.argmin(prev_costs))
        chosen = [None] * len(candidate_sets)
        chosen[-1] = best_last
        for i in range(len(candidate_sets) - 1, 0, -1):
            chosen[i - 1] = backptr[i][chosen[i]]

        optimized = []
        for i, idx in enumerate(chosen):
            q = np.array(candidate_sets[i][idx]['angles'], dtype=float)
            p_fk = np.array(forward_kinematics(q), dtype=float)
            track_error = np.linalg.norm(p_fk - cartesian_points[i])
            if track_error > EE_TRACKING_TOL:
                raise ValueError(
                    f'IK tracking error too high at waypoint {i}: '
                    f'{track_error:.4f}m (tol={EE_TRACKING_TOL:.4f}m).'
                )
            optimized.append((list(q), p_fk))

        set_ik_seed(optimized[-1][0])
        return optimized

    def _build_fast_joint_path(self, cartesian_points, elbow_mode='auto'):
        """
        Fast planner for dense trajectories using sequential warm-started IK.
        
        This method:
          1. Solves first waypoint with full branch search (respects elbow_mode)
          2. Tracks remaining waypoints with warm-start direct IK (140 iterations max)
          3. Falls back to local candidate search if direct solve fails
          4. Validates final position error and tracks time + progress
        
        Returns:
            List of (joint_angles, ee_position) tuples
        
        Complexity:
            Linear in waypoint count; each point has ~0.05-0.2s planning time
        """
        optimized = []
        plan_start = time.perf_counter()
        waypoint_count = len(cartesian_points)

        # First point: global branch selection keeps top/down behavior robust.
        first = cartesian_points[0]
        first_candidates = solve_ik_candidates(
            float(first[0]),
            float(first[1]),
            float(first[2]),
            reference_theta=None,
            elbow_mode=elbow_mode,
            max_solutions=2,
            exhaustive=True,
        )
        if not first_candidates:
            raise ValueError('No IK candidates found at waypoint 0.')

        q_prev = np.array(first_candidates[0]['angles'], dtype=float)
        p_fk = np.array(forward_kinematics(q_prev), dtype=float)
        if np.linalg.norm(p_fk - first) > EE_TRACKING_TOL:
            raise ValueError('Initial waypoint tracking error is too high.')
        optimized.append((list(q_prev), p_fk))

        for i in range(1, waypoint_count):
            if (time.perf_counter() - plan_start) > PLANNING_TIMEOUT_SEC:
                raise ValueError(
                    f'Planning timeout after {PLANNING_TIMEOUT_SEC:.0f}s at waypoint {i}/{waypoint_count}. '
                    f'Reduce speed detail (fewer steps) or choose a simpler path.'
                )

            p = cartesian_points[i]
            try:
                q = solve_ik_direct(
                    float(p[0]),
                    float(p[1]),
                    float(p[2]),
                    reference_theta=q_prev,
                    orientation_weight=0.08,
                    max_iterations=140,
                )
            except ValueError:
                # Local recovery: bounded candidate search around current branch.
                recovered = solve_ik_candidates(
                    float(p[0]),
                    float(p[1]),
                    float(p[2]),
                    reference_theta=q_prev,
                    elbow_mode=elbow_mode,
                    max_solutions=2,
                    exhaustive=False,
                )
                if not recovered:
                    raise ValueError(f'IK failed at waypoint {i}: no recovery candidate found.')
                q = np.array(recovered[0]['angles'], dtype=float)

            p_fk = np.array(forward_kinematics(q), dtype=float)
            track_error = np.linalg.norm(p_fk - p)
            if track_error > EE_TRACKING_TOL:
                raise ValueError(
                    f'IK tracking error too high at waypoint {i}: '
                    f'{track_error:.4f}m (tol={EE_TRACKING_TOL:.4f}m).'
                )

            optimized.append((list(q), p_fk))
            q_prev = q

            if i % 30 == 0:
                elapsed = time.perf_counter() - plan_start
                self.send_status(0, f'Planning progress: {i}/{waypoint_count} waypoints ({elapsed:.1f}s).')

        set_ik_seed(optimized[-1][0])
        return optimized

    # ------------------------------------------------------------------ #
    #  GUI callbacks
    # ------------------------------------------------------------------ #
    def handle_straight(self, msg):
        if len(msg.data) < 7:
            self.send_status(2, 'Straight-line command must contain 7 values: Ax Ay Az Bx By Bz speed.')
            return
        A = list(msg.data[0:3])
        B = list(msg.data[3:6])
        speed = float(msg.data[6])
        elbow_mode = 'auto'
        if len(msg.data) >= 8:
            elbow_mode = self._decode_elbow_mode(float(msg.data[7]))
        if speed <= 0.0:
            self.send_status(2, f'Speed must be positive, got speed={speed}. Aborting.')
            return
        steps = self._adaptive_steps_for_line(A, B, speed)
        self.send_status(0, f'Adaptive straight-line sampling: speed={speed:.2f} -> {steps} steps.')
        self.load_straight(A, B, steps=steps, elbow_mode=elbow_mode)

    def handle_circle(self, msg):
        if len(msg.data) < 5:
            self.send_status(2, 'Circle command must contain 5 values: Cx Cy Cz radius speed_or_steps.')
            return
        center = list(msg.data[0:3])
        R = float(msg.data[3])
        speed_or_steps = float(msg.data[4])
        if speed_or_steps <= 0.0:
            self.send_status(2, f'Circle speed/steps must be positive, got {speed_or_steps}. Aborting.')
            return

        # Backward compatible parsing:
        #  - <= 20 means speed hint from GUI
        #  - > 20 means explicit legacy steps
        if speed_or_steps <= 20.0:
            speed = speed_or_steps
            steps = self._adaptive_steps_for_circle(R, speed)
            self.send_status(0, f'Adaptive circle sampling: speed={speed:.2f}, R={R:.3f}m -> {steps} steps.')
        else:
            requested_steps = int(speed_or_steps)
            steps = int(np.clip(max(10, requested_steps), 10, MAX_TRAJECTORY_POINTS))
            if steps != requested_steps:
                self.send_status(
                    1,
                    f'Circle steps capped from {requested_steps} to {steps} to keep planning responsive.',
                )
        elbow_mode = 'auto'
        if len(msg.data) >= 6:
            elbow_mode = self._decode_elbow_mode(float(msg.data[5]))
        if R <= 0.0:
            self.send_status(2, f'Radius must be positive, got R={R}. Aborting.')
            return
        self.load_circle(center, R, steps=steps, elbow_mode=elbow_mode)

    # ------------------------------------------------------------------ #
    #  Waypoint loading — stores (angles, cartesian) pairs
    # ------------------------------------------------------------------ #
    def load_straight(self, A, B, steps=200, elbow_mode='auto'):
        if not is_reachable(A):
            self.send_status(2,
                f'Point A={A} outside workspace '
                f'(norm={np.linalg.norm(A):.3f}m, allowed 0.2–1.285m). Aborting.')
            return
        if not is_reachable(B):
            self.send_status(2,
                f'Point B={B} outside workspace '
                f'(norm={np.linalg.norm(B):.3f}m, allowed 0.2–1.285m). Aborting.')
            return

        if A[2] < MIN_Z:
            self.send_status(2, f'Point A Z={A[2]:.3f}m too low. Minimum Z={MIN_Z}m. Aborting.')
            return
        if B[2] < MIN_Z:
            self.send_status(2, f'Point B Z={B[2]:.3f}m too low. Minimum Z={MIN_Z}m. Aborting.')
            return

        self.send_status(0, 'Computing straight-line trajectory, please wait...')
        self.loading = True
        self.current_step = 0
        self.waypoints = []
        reset_ik_seed()

        duration = max(float(steps) * self.control_period, 1.0)
        time_vector, cartesian_waypoints, cartesian_velocities, cartesian_accelerations = (
            self._sample_straight_trajectory(A, B, steps, duration)
        )
        self.cartesian_trajectory = list(zip(
            time_vector,
            cartesian_waypoints,
            cartesian_velocities,
            cartesian_accelerations,
        ))

        for i, p in enumerate(cartesian_waypoints):
            if not is_reachable(p):
                self.loading = False
                self.send_status(2,
                    f'Waypoint {i} at {np.round(p,3)} is outside workspace. '
                    f'Choose closer A and B. Aborting.')
                return

        try:
            candidate_waypoints = self._build_optimal_joint_path(
                cartesian_waypoints,
                elbow_mode=elbow_mode,
            )
        except ValueError as exc:
            self.loading = False
            self.send_status(2, f'{exc} Aborting.')
            return

        self.clear_markers()
        self.waypoints = candidate_waypoints
        self.current_step = 0
        self.last_dot_waypoint = -1
        self.last_sample_point = None
        self._prepare_motion_profile()
        self.motion_start_time = time.perf_counter()
        self.motion_active = True
        self.loading = False
        self.send_status(3,
            f'Straight-line ready: {len(self.waypoints)} waypoints, '
            f'estimated duration {self.total_motion_time:.2f}s.')

    def load_circle(self, center, R, steps=300, elbow_mode='auto'):
        cx, cy, cz = center
        rim = [cx + R, cy, cz]

        if not is_reachable(center):
            self.send_status(2,
                f'Center={center} outside workspace '
                f'(norm={np.linalg.norm(center):.3f}m). Aborting.')
            return
        if not is_reachable(rim):
            self.send_status(2,
                f'Circle rim exceeds workspace '
                f'(rim norm={np.linalg.norm(rim):.3f}m, max=1.285m). Aborting.')
            return
        if R <= 0:
            self.send_status(2, f'Radius must be positive, got R={R}. Aborting.')
            return

        if center[2] < MIN_Z:
            self.send_status(2, f'Center Z={center[2]:.3f}m too low. Minimum Z={MIN_Z}m. Aborting.')
            return

        self.send_status(0, 'Computing circular trajectory, please wait...')
        self.loading = True
        self.current_step = 0
        self.waypoints = []
        reset_ik_seed()

        duration = max(float(steps) * self.control_period, 1.0)
        time_vector, cartesian_waypoints, cartesian_velocities, cartesian_accelerations = (
            self._sample_circle_trajectory(center, R, steps, duration)
        )
        self.cartesian_trajectory = list(zip(
            time_vector,
            cartesian_waypoints,
            cartesian_velocities,
            cartesian_accelerations,
        ))

        for i, p in enumerate(cartesian_waypoints):
            if not is_reachable(p):
                self.loading = False
                self.send_status(2,
                    f'Circle point {i} at {np.round(p,3)} outside workspace. '
                    f'Reduce radius or move center. Aborting.')
                return

        try:
            candidate_waypoints = self._build_optimal_joint_path(
                cartesian_waypoints,
                elbow_mode=elbow_mode,
            )
        except ValueError as exc:
            self.loading = False
            self.send_status(2, f'{exc} Aborting.')
            return

        self.clear_markers()
        self.waypoints = candidate_waypoints
        self.current_step = 0
        self.last_dot_waypoint = -1
        self.last_sample_point = None
        self._prepare_motion_profile()
        self.motion_start_time = time.perf_counter()
        self.motion_active = True
        self.loading = False
        self.send_status(3,
            f'Circle ready: {len(self.waypoints)} waypoints, '
            f'estimated duration {self.total_motion_time:.2f}s.')

    # ------------------------------------------------------------------ #
    #  Timer callback
    # ------------------------------------------------------------------ #
    def publish_step(self):
        """
        Timer callback for real-time motion execution.
        
        Implements time-parameterized interpolation across waypoints:
          - Computes elapsed time since trajectory start
          - Finds current segment and interpolation alpha
          - Interpolates joint angles with proper angle wrapping
          - Splines EE position (Cartesian)
          - Updates trace and waypoint markers with adaptive sampling
        
        Joint motion respects velocity/acceleration limits set during planning.
        """
        if self.loading:
            return

        if not self.motion_active or len(self.waypoints) == 0:
            return

        if len(self.waypoints) == 1:
            angles, ee_position = self.waypoints[0]
            self.publish_joints(angles)
            self.publish_trace(*ee_position)
            self.publish_waypoint_dot(*ee_position)
            self.motion_active = False
            self.send_status(3, 'Trajectory complete. Ready for next command.')
            return

        elapsed = time.perf_counter() - self.motion_start_time

        if elapsed >= self.total_motion_time:
            angles, ee_position = self.waypoints[-1]
            self.publish_joints(angles)
            if self.last_sample_point is None or np.linalg.norm(np.array(ee_position) - self.last_sample_point) > 1e-6:
                self.publish_trace(*ee_position)
                self.last_sample_point = np.array(ee_position, dtype=float)
            self.publish_waypoint_dot(*ee_position)
            self.motion_active = False
            self.current_step = len(self.waypoints)
            self.send_status(3, 'Trajectory complete. Ready for next command.')
            return

        seg_idx = int(np.searchsorted(self.waypoint_times, elapsed, side='right') - 1)
        seg_idx = max(0, min(seg_idx, len(self.waypoints) - 2))

        t0 = self.waypoint_times[seg_idx]
        t1 = self.waypoint_times[seg_idx + 1]
        alpha = 0.0 if t1 <= t0 else (elapsed - t0) / (t1 - t0)
        alpha = float(np.clip(alpha, 0.0, 1.0))

        q0 = np.array(self.waypoints[seg_idx][0], dtype=float)
        q1 = np.array(self.waypoints[seg_idx + 1][0], dtype=float)
        dq = self._shortest_joint_delta(q0, q1)
        q_interp = (q0 + alpha * dq + np.pi) % (2.0 * np.pi) - np.pi

        p0 = np.array(self.waypoints[seg_idx][1], dtype=float)
        p1 = np.array(self.waypoints[seg_idx + 1][1], dtype=float)
        p_interp = (1.0 - alpha) * p0 + alpha * p1

        while self.last_dot_waypoint < seg_idx:
            self.last_dot_waypoint += 1
            wp = self.waypoints[self.last_dot_waypoint][1]
            self.publish_waypoint_dot(*wp)

        self.publish_joints(q_interp)
        if self.last_sample_point is None or np.linalg.norm(p_interp - self.last_sample_point) > 1e-4:
            self.publish_trace(*p_interp)
            self.last_sample_point = p_interp

    # ------------------------------------------------------------------ #
    #  Publishers
    # ------------------------------------------------------------------ #
    def publish_joints(self, angles):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = list(angles)
        self.js_pub.publish(msg)

    def publish_trace(self, x, y, z):
        self.trace_points.append(Point(x=float(x), y=float(y), z=float(z)))
        m = Marker()
        m.header.frame_id = 'base_link'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'ee_trace'
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.005
        m.color.r = 1.0
        m.color.a = 1.0
        m.points = self.trace_points
        self.marker_pub.publish(m)

    def publish_waypoint_dot(self, x, y, z):
        ma = MarkerArray()
        m = Marker()
        m.header.frame_id = 'base_link'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = 'waypoints'
        m.id = self.waypoint_id
        self.waypoint_id += 1
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(x)
        m.pose.position.y = float(y)
        m.pose.position.z = float(z)
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.02
        m.color.g = 1.0
        m.color.a = 1.0
        ma.markers.append(m)
        self.marker_array_pub.publish(ma)


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
