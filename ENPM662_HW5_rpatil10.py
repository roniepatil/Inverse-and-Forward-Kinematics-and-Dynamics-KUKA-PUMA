# ENPM662 Homework 5
# Author: Rohit M Patil
# Email: rpatil10@umd.edu
# Due Date: Dec 1

#### Import libraries ####
from operator import inv, sub
from sympy import *
from sympy.plotting.textplot import linspace
import numpy as np
import matplotlib.pyplot as plt

#### Tranformation Matrix of KUKA Robot ####
def tfMatrix(t_i, al_i, d_i, a_i):
    R_theta = Matrix([[cos(t_i), -sin(t_i), 0, a_i],[sin(t_i), cos(t_i), 0, 0],[0, 0, 1, d_i],[0, 0, 0, 1]])
    R_alpha = Matrix([[1, 0, 0, 0],[0, cos(al_i), -sin(al_i), 0],[0, sin(al_i), cos(al_i), 0],[0, 0, 0, 1]])
    T = R_theta*R_alpha
    return T

q1, q2, q4, q5, q6, q7 = symbols('theta1 theta2 theta4 theta5 theta6 theta7')

T12 = tfMatrix(q1, -pi/2, 360, 0)
T23 = tfMatrix(q2, pi/2, 0, 0)
T34 = tfMatrix(0, pi/2, 420, 0)
T45 = tfMatrix(q4, -pi/2, 0, 0)
T56 = tfMatrix(q5, -pi/2, 399.5, 0)
T67 = tfMatrix(q6, pi/2, 0, 0)
T7eff = tfMatrix(q7, 0, 205.5, 0)

T1eff = T12*T23*T34*T45*T56*T67*T7eff

T13 = T12*T23
T14 = T13*T34
T15 = T14*T45
T16 = T15*T56
T17 = T12*T23*T34*T45*T56*T67

#### Heights of links ####
h1 = 0
h2 = (T12[2, 3] + h1)/2
h3 = (T13[2, 3] + T12[2, 3])/2
h4 = (T14[2, 3] + T13[2, 3])/2
h5 = (T15[2, 3] + T14[2, 3])/2
h6 = (T16[2, 3] + T15[2, 3])/2
h7 = (T17[2, 3] + T16[2, 3])/2

#### Potential energy ####
P1 = diff(h1, q2) + diff(h1, q4)
P2 = diff(h2, q2) + diff(h2, q4)
P4 = diff(h4, q2) + diff(h4, q4)
P5 = diff(h5, q2) + diff(h5, q4)
P6 = diff(h6, q2) + diff(h6, q4)
P7 = diff(h7, q2) + diff(h7, q4)
P = Matrix([P1, P2, P4, P5, P6, P7])

#### Mass of KUKA from datasheet ####
mass = 22.3

#### Acceleration due Gravity ####
g = 9.8
G = (Symbol("m")/4)*Symbol("g")*P/1000

#### Find Jacobian ####
def Jacobian(theta1, theta2, theta4, theta5, theta6, thata7):
    T12 = tfMatrix(theta1, -pi/2, 360, 0)
    T23 = tfMatrix(theta2, pi/2, 0, 0)
    T34 = tfMatrix(0, pi/2, 420, 0)
    T45 = tfMatrix(theta4, -pi/2, 0, 0)
    T56 = tfMatrix(theta5, -pi/2, 399.5, 0)
    T67 = tfMatrix(theta6, pi/2, 0, 0)
    T7eff = tfMatrix(thata7, 0, 205.5, 0)
    T1eff = T12*T23*T34*T45*T56*T67*T7eff
    T13 = T12*T23
    T14 = T13*T34
    T15 = T14*T45
    T16 = T15*T56
    T17 = T12*T23*T34*T45*T56*T67

    z0 = Matrix([[0], [0], [1]])
    z1 = T12[0:3, 2]
    z2 = T13[0:3, 2]
    z3 = T14[0:3, 2]
    z4 = T15[0:3, 2]
    z5 = T16[0:3, 2]
    z6 = T17[0:3, 2]
    z7 = T1eff[0:3, 2]

    o0 = Matrix([[0],[0],[0]])
    o1 = T12[0:3, 3]
    o2 = T13[0:3, 3]
    o3 = T14[0:3, 3]
    o4 = T15[0:3, 3]
    o5 = T16[0:3, 3]
    o6 = T17[0:3, 3]
    o7 = T1eff[0:3, 3]

    jv1 = z0.cross(o7 - o0)
    j1 = Matrix([[(jv1)],[(z0)]])
    jv2 = z1.cross(o7 - o1)
    j2 = Matrix([[(jv2)],[(z1)]])
    jv3 = z2.cross(o7 - o2)
    j3 = Matrix([[(jv3)],[(z2)]])
    jv4 = z3.cross(o7 - o3)
    j4 = Matrix([[(jv4)],[(z3)]])
    jv5 = z4.cross(o7-o4)
    j5 = Matrix([[(jv5)],[(z4)]])
    jv6 = z5.cross(o7 - o5)
    j6 = Matrix([[(jv6)],[(z5)]])
    jv7 = z6.cross(o7 - o6)
    j7 = Matrix([[(jv7)],[(z6)]])
    j = Matrix([[j1, j2, j4, j5, j6, j7]])
    return j


N = 50  
q = np.zeros((6, N))  
t = linspace(0, 200, num=N)

#### Starting conditions ####
q[0, 0] = 0
q[1, 0] = -pi/6
q[2, 0] = -pi/4
q[3, 0] = 0
q[4, 0] = 5*pi/12
q[5, 0] = 0
FT = np.zeros((6, N))

#### Iterate N times ####
for i in range(0, N-1):

    x = -100*sin(2*i*pi/N)*2*pi/200 
    y = 100*cos(2*i*pi/N)*2*pi/200
    v = Matrix([x, y, 0, 0, 0, 0])
    J = Jacobian(q[0, i], q[1, i],q[2, i], q[3, i], q[4, i], q[5, i])
    Q_dot = Inverse(J)*v
    F = Transpose(J)*Matrix([0, 0, 5, 0, 0, 0])
    GM = Matrix([0, 0, -21*mass*g*sin(q[2, i])/400, -21*mass*g *sin(q[2, i])/200, -21*mass*g*sin(q[2, i])/200, -21*mass*g*sin(q[2, i])/200])
    
    FT[0, i] = F[0, 0] + GM[0, 0]
    FT[1, i] = F[1, 0] + GM[1, 0]
    FT[2, i] = F[2, 0] + GM[2, 0]
    FT[3, i] = F[3, 0] + GM[3, 0]
    FT[4, i] = F[4, 0] + GM[4, 0]
    FT[5, i] = F[5, 0] + GM[5, 0]

    q[0, i+1] = q[0, i] + Q_dot[0, 0]*(200/N)
    q[1, i+1] = q[1, i] + Q_dot[1, 0]*(200/N)
    q[2, i+1] = q[2, i] + Q_dot[2, 0]*(200/N)
    q[3, i+1] = q[3, i] + Q_dot[3, 0]*(200/N)
    q[4, i+1] = q[4, i] + Q_dot[4, 0]*(200/N)
    q[5, i+1] = q[5, i] + Q_dot[5, 0]*(200/N)

#### Ploting the joint torques ####
plt.xlabel('Time(sec)')
plt.ylabel('Torque(N mm)')
plt.title('Joint 1')
plt.scatter(t, FT[0, :], color='blue')
plt.show()
plt.xlabel('Time(sec)')
plt.ylabel('Torque(N mm)')
plt.title('Joint 2')
plt.scatter(t, FT[1, :], color='blue')
plt.show()
plt.xlabel('Time(sec)')
plt.ylabel('Torque(N mm)')
plt.title('Joint 4')
plt.scatter(t, FT[2, :], color='blue')
plt.show()
plt.xlabel('Time(sec)')
plt.ylabel('Torque(N mm)')
plt.title('Joint 5')
plt.scatter(t, FT[3, :], color='blue')
plt.show()
plt.xlabel('Time(sec)')
plt.ylabel('Torque(N mm)')
plt.title('Joint 6')
plt.scatter(t, FT[4, :], color='blue')
plt.show()
plt.xlabel('Time(sec)')
plt.ylabel('Torque(N mm)')
plt.title('Joint 7')
plt.scatter(t, FT[5, :], color='blue')
plt.show()