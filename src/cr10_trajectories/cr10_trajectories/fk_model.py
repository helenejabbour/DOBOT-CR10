#!/usr/bin/python3
"""
fk_model.py

Symbolic forward kinematics and Jacobian for Dobot CR10.

This module uses SymPy to build the kinematic chain symbolically, then compiles
it to fast NumPy functions via lambdify for repeated evaluation during trajectory planning.

DH Convention:
  Standard DH with link offsets applied to joints 2 and 4 to match robot calibration:
    θ2_actual = θ2_cmd + π/2
    θ4_actual = θ4_cmd - π/2

DH Parameters (CR10, in meters):
  Link 0→1: d1=0.1765,  a=0,    α=π/2
  Link 1→2: d=0,        a2=0.607, α=0   (+ θ offset)
  Link 2→3: d=0,        a3=0.568, α=0
  Link 3→4: d4=0.193,   a=0,    α=-π/2 (- θ offset)
  Link 4→5: d5=0.125,   a=0,    α=π/2
  Link 5→6: d6=0.1114,  a=0,    α=0

Public API:
  - get_fk(q): Returns 4×4 homogeneous transform T06 for joint config q
  - get_jacobian(q): Returns 6×6 geometric Jacobian (position + orientation)
  - get_ee_position(q): Returns (x, y, z) end-effector position

Internals:
  - fk_func, jac_func: Compiled lambdify expressions (fast evaluation)
  - T06, J, P: SymPy expressions for the full chain and Jacobian
"""
import numpy as np
from math import pi
from sympy import symbols, Matrix, cos, sin, lambdify

# ======================================================
# 1. CR10 DH PARAMETERS (meters)
# ======================================================
d1 = 0.1765
a2 = 0.607
a3 = 0.568
d4 = 0.193
d5 = 0.125
d6 = 0.1114

# # ======================================================
# # RobotSerial DH PARAMETERS
# #  (NO theta offsets inside DH table)
# # ======================================================
# dh_params = np.array([
#   [d1, 0,  pi/2, 0],
#   [0, a2, 0,  0],
#   [0, a3, 0,  0],
#   [d4, 0, -pi/2, 0],
#   [d5, 0,  pi/2, 0],
#   [d6, 0,  0,  0]
# ])

# robot = RobotSerial(dh_params)

# # ======================================================
# # TEST JOINT CONFIGURATION
# #  (IMPORTANT: apply the SAME θ offsets used symbolically)
# # ======================================================
# theta_raw = np.array([pi/2, 1.3, -pi/4, 0, pi/7, pi/5])

# theta_robot = np.array([
#   theta_raw[0],
#   theta_raw[1] + pi/2,  # θ2 offset
#   theta_raw[2],
#   theta_raw[3] - pi/2,  # θ4 offset
#   theta_raw[4],
#   theta_raw[5]
# ])

# # ======================================================
# # NUMERICAL FK USING RobotSerial
# # ======================================================
# f_numeric = robot.forward(theta_robot)

# print("FK using RobotSerial:")
# print("Position (x,y,z):")
# print(f_numeric.t_3_1.flatten())

# ======================================================
# 2. SYMBOLIC VARIABLES
# ======================================================
theta1, theta2, theta3, theta4, theta5, theta6 = symbols(
  'theta1 theta2 theta3 theta4 theta5 theta6'
)
_syms = [theta1, theta2, theta3, theta4, theta5, theta6]

# ======================================================
# 3. STANDARD DH TRANSFORMATION
# ======================================================
def DH(theta, d, a, alpha):
  return Matrix([
    [cos(theta), -sin(theta)*cos(alpha), sin(theta)*sin(alpha), a*cos(theta)],
    [sin(theta), cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
    [0, sin(alpha), cos(alpha), d],
    [0, 0, 0, 1]
  ])

# ======================================================
# 4. SYMBOLIC FK (CR10 – MATCHES RobotSerial)
# ======================================================
T01 = DH(theta1,    d1, 0, pi/2)
T12 = DH(theta2 + pi/2, 0, a2, 0)
T23 = DH(theta3,    0, a3, 0)
T34 = DH(theta4 - pi/2, d4, 0, -pi/2)
T45 = DH(theta5,    d5, 0, pi/2)
T56 = DH(theta6,    d6, 0, 0)

T02 = T01 * T12
T03 = T02 * T23
T04 = T03 * T34
T05 = T04 * T45
T06 = T05 * T56

# ======================================================
# 5. END-EFFECTOR POSITION
# ======================================================
P = T06[:3, 3]

# ======================================================
# 6. POSITION JACOBIAN
# ======================================================
Jv = Matrix([
  [P[0].diff(theta1), P[0].diff(theta2), P[0].diff(theta3),
   P[0].diff(theta4), P[0].diff(theta5), P[0].diff(theta6)],
  [P[1].diff(theta1), P[1].diff(theta2), P[1].diff(theta3),
   P[1].diff(theta4), P[1].diff(theta5), P[1].diff(theta6)],
  [P[2].diff(theta1), P[2].diff(theta2), P[2].diff(theta3),
   P[2].diff(theta4), P[2].diff(theta5), P[2].diff(theta6)]
])

# ======================================================
# 7. ANGULAR JACOBIAN (Z0 → Z5 ONLY)
# ======================================================
Z0 = Matrix([0, 0, 1])
Z1 = T01[:3, 2]
Z2 = T02[:3, 2]
Z3 = T03[:3, 2]
Z4 = T04[:3, 2]
Z5 = T05[:3, 2]

Jw = Matrix.hstack(Z0, Z1, Z2, Z3, Z4, Z5)

# ======================================================
# 8. FULL GEOMETRIC JACOBIAN
# ======================================================
J = Matrix.vstack(Jv, Jw)

# ======================================================
# 9. NUMERICAL EVALUATION
# lambdify compiles symbolic expressions
#  to numpy functions once — replaces sympy.subs() in loops
# ======================================================
fk_func  = lambdify(_syms, T06, modules='numpy')
jac_func = lambdify(_syms, J,   modules='numpy')

def get_fk(q):
    """Returns 4x4 homogeneous transform for joint config q."""
    return np.array(fk_func(*q), dtype=float)

def get_jacobian(q):
    """Returns 6x6 Jacobian matrix for joint config q."""
    return np.array(jac_func(*q), dtype=float)

def get_ee_position(q):
    """Returns (x, y, z) end-effector position for joint config q."""
    T = get_fk(q)
    return T[0, 3], T[1, 3], T[2, 3]

# ======================================================
# # VALIDATION (run as standalone script)
# # ======================================================
# if __name__ == '__main__':
#     subs = {
#       theta1: theta_raw[0],
#       theta2: theta_raw[1],
#       theta3: theta_raw[2],
#       theta4: theta_raw[3],
#       theta5: theta_raw[4],
#       theta6: theta_raw[5]
#     }

#     T06_num = T06.subs(subs).evalf()
#     P_num = T06_num[:3, 3]
#     J_num = J.subs(subs).evalf()

#     print("\nSymbolic FK result:")
#     print("x =", float(P_num[0]))
#     print("y =", float(P_num[1]))
#     print("z =", float(P_num[2]))

#     print("\nJacobian Matrix:")
#     print(np.array(J_num, dtype=float))