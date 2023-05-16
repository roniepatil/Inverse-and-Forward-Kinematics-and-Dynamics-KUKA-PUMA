import numpy as np
from numpy import array
from sympy import symbols, cos, sin, pi, simplify, sqrt, atan2, pprint
from sympy.matrices import Matrix

# Create symbols for DH param
q1, q2, q3, q4, q5, q6 = symbols('q1:7')                                 # joint angles theta
d1, d2, d3, d4, d5, d6 = symbols('d1:7')                                 # link offsets
a0, a1, a2, a3, a4, a5 = symbols('a0:6')                                 # link lengths
alpha0, alpha1, alpha2, alpha3, alpha4, alpha5 = symbols('alpha0:6') # joint twist angles

# DH Table
dh = {alpha0:      0, a0:      0, d1:     0, q1:        q1,
      alpha1: -pi/2., a1:      0, d2:     0, q2:        q2,
      alpha2:      0, a2:     a2, d3:    d3, q3:        q3,
      alpha3: -pi/2., a3:     a3, d4:    d4, q4:        q4,
      alpha4:  pi/2., a4:      0, d5:     0, q5:        q5,
      alpha5: -pi/2., a5:      0, d6:     0, q6:        q6}

# Function to return homogeneous transform matrix

def TF_Mat(alpha, a, d, q):
    TF = Matrix([[ cos(q), -sin(q)*cos(alpha),  sin(q)*sin(alpha),  a*cos(q)],
                 [ sin(q), cos(q)*cos(alpha),   -cos(q)*sin(alpha), a*sin(q)],
                 [ 0,      sin(alpha),          cos(alpha),         d],
                 [ 0,      0,                   0,                  1]])
    return TF

## Substiute DH_Table
T0_1 = TF_Mat(alpha0, a0, d1, q1).subs(dh)
T1_2 = TF_Mat(alpha1, a1, d2, q2).subs(dh)
T2_3 = TF_Mat(alpha2, a2, d3, q3).subs(dh)
T3_4 = TF_Mat(alpha3, a3, d4, q4).subs(dh)
T4_5 = TF_Mat(alpha4, a4, d5, q5).subs(dh)
T5_6 = TF_Mat(alpha5, a5, d6, q6).subs(dh)

# Composition of Homogeneous Transforms
# Transform from Base link to end effector (Gripper)
T0_2 = (T0_1 * T1_2) ## (Base) Link_0 to Link_2
T0_3 = (T0_2 * T2_3) ## (Base) Link_0 to Link_3
T0_4 = (T0_3 * T3_4) ## (Base) Link_0 to Link_4
T0_5 = (T0_4 * T4_5) ## (Base) Link_0 to Link_5
T0_6 = (T0_5 * T5_6) ## (Base) Link_0 to Link_6 (End Effector)


print("\n")
pprint(T0_1)
pprint(T1_2)
pprint(T2_3)
pprint(T3_4)
pprint(T4_5)
pprint(T5_6)
print("\n")
pprint(T0_1)
pprint(T0_2)
pprint(T0_3)
pprint(T0_4)
pprint(T0_5)
pprint(T0_6)