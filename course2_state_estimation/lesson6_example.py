import numpy as np
import math
import random
import matplotlib.pyplot as plt

## This is a solution to the UKF example with the car/camera visualizing a landmark in Lesson6

# Some of these variables are placeholders, and only used during debugging. There is a lot
# of optimization that can be done to this
## 

# Height of landmark
S = 20.0
# Distance to landmark
D = 40.0
# State vector [position, velocity]
x_k = np.array([0.0, 5.0])
# State uncertainity matrix
P_k = np.array([[0.01, 0], [0.0, 1.0]])
# Process noise
Q_k = np.array([[0.1, 0], [0.0, 0.1]])
# Input control
u = -2.0
# First observation
y_obs = math.pi / 6.0
# Delta T
time_step = 0.5
# Transition matrix
F = np.array([[1.0, time_step], [0.0, 1.0]])
# Control Matrix
G = np.array([0, time_step])
N = np.size(P_k, axis=0)
num_sigma = 2 * N + 1
# Kappa, tuning parameter
k = 3. - N


def compute_sigma_points(x, P):
    L = np.linalg.cholesky(P)
    sigma_points = np.zeros((num_sigma, N))
    # Initialize the first sigma point to the mean, or current best guess
    sigma_points[0, 0] = x[0]
    sigma_points[0, 1] = x[1]
    for i in range(1, N+1):
        # This is the distance from the mean to get the sigma
        # point. 
        distance_from_mean = math.sqrt(N + k) * L[:, i-1]
        # Calculate the sigma points where sigma_points[0, :] is 
        # the mean value already set
        sigma_points[i, :] = sigma_points[0, :] + distance_from_mean
        sigma_points[i + N, :] = sigma_points[0, :] - distance_from_mean
    
    return sigma_points


## Prediction step
# 1. Calculate sigma points 
s = compute_sigma_points(x_k, P_k)
# 2. Run sigma points through prediction non linear transform
sp_mean = np.zeros((2, 1))
for i in range(np.size(s, axis=0)):
    
    f = np.dot(F, s[i, :].reshape(2, 1))
    g = np.dot(G, u)
    s[i, :] = f.reshape(1, 2) + g.reshape(1, 2)

    if i == 0:
        sp_mean[0] = sp_mean[0] + (k / (N + k)) * s[i, 0]
        sp_mean[1] = sp_mean[1] + (k / (N + k)) * s[i, 1] 
    else:
        sp_mean[0] = sp_mean[0] + (1 / (2.0 * (N + k))) * s[i, 0]
        sp_mean[1] = sp_mean[1] + (1 / (2.0 * (N + k))) * s[i, 1] 

sp_cov = np.zeros((2, 2))
for i in range(np.size(s, axis=0)):
    alpha = 0
    if i is 0:
        alpha = k / (N + k)
    else:
        alpha = 0.5 * (1.0 / (N + k))

    # The observation
    y_i = s[i, :].reshape(2, 1)
    # Observation minus mean
    t = y_i - sp_mean.reshape(2, 1)
    
    sp_cov = sp_cov + alpha * np.dot(t, t.transpose())

# Make sure to add process noise
sp_cov = sp_cov + Q_k
# Remember the mean and cov of sigma points now is our best guess, so set
# P_k and x_k
x_k = sp_mean.reshape(2, 1)
P_k = sp_cov
# Correction step
s = compute_sigma_points(x_k, P_k)
# Measurement transform the sigma points and compute the
# weighted mean of the measurement transform
y = np.zeros((np.size(s, axis=0), 1))
y_mean = 0
for i in range(np.size(s, axis=0)):
    alpha = 0
    if i is 0:
        alpha = k / (N + k)
    else:
        alpha = 0.5 * (1.0 / (N + k))
    
    y[i] = math.atan(S / (D - s[i, 0]))
    y_mean = y_mean + alpha * y[i]

y_cov = 0
for i in range(np.size(s, axis=0)):
    alpha = 0
    if i is 0:
        alpha = k / (N + k)
    else:
        alpha = 0.5 * (1.0 / (N + k))
    
    y[i] = math.atan(S / (D - s[i, 0]))
    y_cov = y_cov + alpha * (y[i] - y_mean)**2

# Make sure to add measurement noise
y_cov = y_cov + 0.01

P_xy = np.zeros((2, 1))
for i in range(np.size(s, axis=0)):
    alpha = 0
    if i is 0:
        alpha = k / (N + k)
    else:
        alpha = 0.5 * (1.0 / (N + k))
    
    P_xy = P_xy + (alpha * (s[i, :].reshape(2, 1) - x_k) * (y[i] - y_mean)).reshape(2, 1)

K = P_xy * (1.0 / y_cov)
x_k = x_k + K * (y_obs - y_mean)

print(x_k)
