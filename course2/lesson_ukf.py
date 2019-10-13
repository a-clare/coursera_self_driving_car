import numpy as np
import math
import random
import matplotlib.pyplot as plt

def plot_ellipse(figAx, mean, std, c):
    ellipse_range = np.linspace(0, 2 * math.pi, 100)

    figAx.plot(mean[0] + std[0] * np.cos(ellipse_range), \
        mean[1] + std[1] * np.sin(ellipse_range), c+'-')
    
    figAx.plot(mean[0], mean[1], c+'*', markersize=12)

num_meas = 500
# Observation array where the columns are r and theta
obs = np.zeros((num_meas, 2))
# Transformed observations into xy coordinates
obs_xy = np.zeros((num_meas, 2))

for i in range(num_meas):
    obs[i, 0] = random.uniform(0.990, 1.010)
    obs[i, 1] = random.uniform(1.2, 1.9)

    obs_xy[i, 0] = obs[i, 0] * math.cos(obs[i, 1])
    obs_xy[i, 1] = obs[i, 0] * math.sin(obs[i, 1])

mean_obs = np.mean(obs, axis=0)
mean_obs_xy = np.mean(obs_xy, axis=0)
std_obs = np.std(obs, axis=0)
std_obs_xy = np.std(obs_xy, axis=0)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(obs[:, 0], obs[:, 1], 'b.')
plot_ellipse(ax1, mean_obs, std_obs, 'g')
ax1.set(xlabel='r', ylabel=r"$\Theta$", title='Polar')
ax2.plot(obs_xy[:, 0], obs_xy[:, 1], 'b.')
plot_ellipse(ax2, mean_obs_xy, std_obs_xy, 'g')
ax2.set(xlabel='x', ylabel='y', title='XY')
plt.grid(True)
plt.savefig('polar_vs_xy_data_comparsion.png', dpt=300)
plt.show()

# Linearized transform
obs_xy_lin = np.zeros((num_meas, 2))
for i in range(num_meas):

    obs_xy_lin[i, 0] = mean_obs[0] * math.cos(mean_obs[1]) + \
        math.cos(mean_obs[1]) * (obs[i, 0] - mean_obs[0]) - \
        mean_obs[0] * math.sin(mean_obs[1]) * (obs[i, 1] - mean_obs[1])
    obs_xy_lin[i, 1] = mean_obs[0] * math.sin(mean_obs[1]) + \
        math.sin(mean_obs[1]) * (obs[i, 0] - mean_obs[0]) + \
        mean_obs[0] * math.cos(mean_obs[1]) * (obs[i, 1] - mean_obs[1])

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(obs[:, 0], obs[:, 1], 'b.')
plot_ellipse(ax1, mean_obs, std_obs, 'g')
ax1.set(xlabel='r', ylabel=r"$\Theta$", title='Polar')
ax2.plot(obs_xy[:, 0], obs_xy[:, 1], 'b.')
ax2.plot(obs_xy_lin[:, 0], obs_xy_lin[:, 1], 'm.')
plot_ellipse(ax2, mean_obs_xy, std_obs_xy, 'g')
ax2.set(xlabel='x', ylabel='y', title='XY')
plt.savefig('polar_vs_xs_vs_linear_transformed.png', dpt=300)
plt.show()


# Unscented Kalman Filter implementation
# Initialize the state uncertainity, or covariance matrix
# This is the uncertainity or variances for our observations which
# are r and theta
P = np.array([[std_obs[0]**2, 0.0], [0.0, std_obs[1]**2]])
# Perform cholesky decomp to get L matrix
L = np.linalg.cholesky(P)
# We are in a 2 dimensional situation, r and theta
N = 2
num_sigma = 2 * N + 1
# Kappa, tuning parameter
k = 2 - N
# Initialize the sigma point vector.
sigma_points = np.zeros((num_sigma, N))
# The first sigma point is the mean before the transformation
# Following same convention where 1st column is r and 2nd column
# is theta
sigma_points[0, 0] = mean_obs[0]
sigma_points[0, 1] = mean_obs[1]
# We need num_sigma number of points. However the first one is 
# already initialized to the mean, and the two sets of sigma points
# are symetrical (+-) so only need to loop N (number of dimensions)
# Need to make sure to keep the loop from 1 to N, not 0 to less than N
# which is the traditional python loop
for i in range(1,N+1):
    # This is the distance from the mean to get the sigma
    # point. 
    distance_from_mean = math.sqrt(N + k) * L[:, i-1]
    # Calculate the sigma points where sigma_points[0, :] is 
    # the mean value already set
    sigma_points[i, :] = sigma_points[0, :] + distance_from_mean
    sigma_points[i + N, :] = sigma_points[0, :] - distance_from_mean


# Need to pass the sigma points through the nonlinear transform
sigma_points_transformed = np.zeros((num_sigma, N))
for i in range(np.size(sigma_points, 0)):
    # x = r * cos(theta)
    sigma_points_transformed[i, 0] = sigma_points[i, 0] * math.cos(sigma_points[i, 1])
    # y = r * sin(theta)
    sigma_points_transformed[i, 1] = sigma_points[i, 0] * math.sin(sigma_points[i, 1])
   
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(obs[:, 0], obs[:, 1], 'b.')
plot_ellipse(ax1, mean_obs, std_obs, 'g')
ax1.plot(sigma_points[:, 0], sigma_points[:, 1], 'y*', markersize=12)
ax1.set(xlabel='r', ylabel=r"$\Theta$", title='Polar')
ax2.plot(obs_xy[:, 0], obs_xy[:, 1], 'b.')
plot_ellipse(ax2, mean_obs_xy, std_obs_xy, 'g')
ax2.plot(sigma_points_transformed[:, 0], sigma_points_transformed[:, 1], 'y*', markersize=12)
ax2.set(xlabel='x', ylabel='y', title='XY')
plt.savefig('calculated_sigma_points.png', dpt=300)
plt.show()

# Need to now compute the mean and covariance for the transformed points
# Going to use spt for sigma points transformed
# Remember N == 2, so this is a 2x1 and 2x2 matrix
# Also remember we are now in the XY dimension after the transform
spt_mean = np.zeros((N, 1))
spt_cov = np.zeros((N, N))

# First calculate the mean
for i in range(num_sigma):
    alpha = 0
    if i is 0:
        alpha = k / (N + k)
    else:
        alpha = 0.5 * (1.0 / N + k)

    # X mean
    spt_mean[0] = spt_mean[0] + alpha * sigma_points_transformed[i, 0]
    # Y mean
    spt_mean[1] = spt_mean[1] + alpha * sigma_points_transformed[i, 1]

for i in range(np.size(sigma_points_transformed, 0)):
    alpha = 0
    if i is 0:
        alpha = k / (N + k)
    else:
        alpha = 0.5 * (1.0 / N + k)
    
    # The observation
    y_i = sigma_points_transformed[i, :].reshape(2, 1)
    # Observation minus mean
    t = y_i - spt_mean.reshape(2, 1)
    
    spt_cov = spt_cov + alpha * np.dot(t, t.transpose())


spt_std = np.array([math.sqrt(spt_cov[0, 0]), math.sqrt(spt_cov[1, 1])])

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(obs[:, 0], obs[:, 1], 'b.')
plot_ellipse(ax1, mean_obs, std_obs, 'g')
ax1.plot(sigma_points[:, 0], sigma_points[:, 1], 'y*', markersize=12)
ax1.set(xlabel='r', ylabel=r"$\Theta$", title='Polar')
ax2.plot(obs_xy[:, 0], obs_xy[:, 1], 'b.')
plot_ellipse(ax2, mean_obs_xy, std_obs_xy, 'g')
ax2.plot(sigma_points_transformed[:, 0], sigma_points_transformed[:, 1], 'y*', markersize=12)
ax2.plot(spt_mean[0], spt_mean[1], 'r*', markersize=12)
plot_ellipse(ax2, spt_mean, spt_std, 'r')
ax2.set(xlabel='x', ylabel='y', title='XY')
plt.savefig('final_comparison_unscented_vs_real.png', dpt=300)
plt.show()