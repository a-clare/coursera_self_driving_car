import numpy as np
from numpy.linalg import inv
import math as math

# Going nonlinear - The Extended Kalman Filter
# This is the answer to the quiz in the video 

# Height of landmark
S = 20.0
# Distance to landmark
D = 40.0
# State vector [position, velocity]
x_k = np.array([0.0, 5.0])
# State uncertainity matrix
P_k = np.array([[0.01, 0], [0.0, 1.0]])
# Input control
u = -2.0
# First observation
y = math.pi / 6.0
# Delta T
time_step = 0.5
# Transition matrix
F = np.array([[1.0, time_step], [0.0, 1.0]])
# Control Matrix
G = np.array([0, time_step])

# Measurement noise
meas_noise = 0.01
# Process model noise
Q = np.array([[0.1, 0], [0, 0.1]])

# Prediction Step
# M and L are 1, so can ignore them
x_k = np.dot(F, x_k.T) + np.dot(G.T, u)
P_k = np.dot(np.dot(F, P_k), F.T) + Q
# Jacobian
H = np.array([S / ((D - x_k[0])**2 + S**2), 0.0])
# Gain
temp = np.dot(P_k, H.T)
temp2 = np.dot(H, temp)
temp2 = 1.0 / (temp2 + meas_noise)
K = np.dot(temp, temp2)
# Innovation
z = y - math.atan((S / (D - x_k[0])))
x_k = x_k + np.dot(K, z)
P_k = np.dot(np.eye(2, 2) - np.dot(K, H), P_k)

print(x_k)
print(P_k)