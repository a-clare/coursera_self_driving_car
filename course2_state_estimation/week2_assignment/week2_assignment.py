import pickle
import numpy as np
import matplotlib.pyplot as plt
from math import *

def wraptopi(x):
    if x > np.pi:
        x = x - (np.floor(x / (2 * np.pi)) + 1) * 2 * np.pi
    elif x < -np.pi:
        x = x + (np.floor(x / (-2 * np.pi)) + 1) * 2 * np.pi
    return x

# lk = landmark x,y
# rk = range measurements
# bk = bearing to each landmark
#
#
def measurement_update(lk, rk, bk, P_check, x_check):
    
    # 1. Compute measurement Jacobian

    # 2. Compute Kalman Gain

    # 3. Correct predicted state (remember to wrap the angles to [-pi,pi])

    # 4. Correct covariance

    return x_check, P_check


with open('week2_assignment/data.pickle', 'rb') as f:
    data = pickle.load(f)

t = data['t']  # timestamps [s]

x_init  = data['x_init'] # initial x position [m]
y_init  = data['y_init'] # initial y position [m]
th_init = data['th_init'] # initial theta position [rad]

# input signal
v  = data['v']  # translational velocity input [m/s]
om = data['om']  # rotational velocity input [rad/s]

# bearing and range measurements, LIDAR constants
b = data['b']  # bearing to each landmarks center in the frame attached to the laser [rad]
r = data['r']  # range measurements [m]
l = data['l']  # x,y positions of landmarks [m]
d = data['d']  # distance between robot center and laser rangefinder [m]

v_var = 0.01  # translation velocity variance  
om_var = 0.01  # rotational velocity variance 
r_var = 0.1  # range measurements variance
b_var = 0.1  # bearing measurement variance

Q_km = np.diag([v_var, om_var]) # input noise covariance 
cov_y = np.diag([r_var, b_var])  # measurement noise covariance 

x_est = np.zeros([len(v), 3])  # estimated states, x, y, and theta
P_est = np.zeros([len(v), 3, 3])  # state covariance matrices

x_est[0] = np.array([x_init, y_init, th_init]) # initial state
P_est[0] = np.diag([1, 1, 0.1]) # initial state covariance
# Initialze the state uncertainity matrix
P_check = P_est[0]

for k in range(1, len(t)):  # start at 1 because we've set the initial prediciton

    delta_t = t[k] - t[k - 1]  # time step (difference between timestamps)
    
    # 1. Update state with odometry readings (remember to wrap the angles to [-pi,pi])
    # This is the state vector at the previous step
    x_check = x_est[k - 1].reshape(3, 1)
    # Break out theta before update since its used in several spots
    theta = x_check[2]
    sin_theta_delta_t = sin(theta) * delta_t
    cos_theta_delta_t = cos(theta) * delta_t
    # Propogate the state forward to the current step
    x_check[0] += cos_theta_delta_t * v[k]
    x_check[1] += sin_theta_delta_t * v[k]
    x_check[2] += delta_t * om[k]
    #x_check[2] = wraptopi(x_check[2])
    
    # 2. Motion model jacobian with respect to last state
    F_km = np.array([
        [1.0, 0.0, -sin_theta_delta_t * v[k]],
        [0.0, 1.0, cos_theta_delta_t * v[k]],
        [0.0, 0.0, 1.0]])

    # 3. Motion model jacobian with respect to noise
    L_km = np.array([
        [cos_theta_delta_t, 0.0],
        [sin_theta_delta_t, 0.0],
        [0.0, 1.0]])
    # 4. Propagate uncertainty
    P_check = F_km @ P_check @ F_km.T + L_km @ Q_km @ L_km.T

    # 5. Update state estimate using available landmark measurements
    for i in range(len(r[k])):
        x_check, P_check = measurement_update(l[i], r[k, i], b[k, i], P_check, x_check)

    # Set final state predictions for timestep
    x_est[k, 0] = x_check[0]
    x_est[k, 1] = x_check[1]
    x_est[k, 2] = x_check[2]
    P_est[k, :, :] = P_check

e_fig = plt.figure()
ax = e_fig.add_subplot(111)
ax.plot(x_est[:, 0], x_est[:, 1])
plt.plot(x_init, y_init, 'r*')
plt.plot(x_est[-1, 0], x_est[-1, 1], 'r^')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Estimated trajectory')
plt.savefig('images/week2_trajectory_xy_no_update_step.png', dpt=300)
plt.show()

e_fig = plt.figure()
ax = e_fig.add_subplot(111)
ax.plot(t[:], x_est[:, 2])
ax.set_xlabel('Time [s]')
ax.set_ylabel('theta [rad]')
ax.set_title('Estimated trajectory')
plt.savefig('images/week2_trajectory_time_no_update_step.png', dpt=300)
plt.show()


std_x = np.zeros((len(t), 1))
std_y = np.zeros((len(t), 1))
std_theta = np.zeros((len(t), 1))
for i in range(len(t)):
    std_x[i] = sqrt(P_est[i, 0, 0])
    std_y[i] = sqrt(P_est[i, 1, 1])
    std_theta[i] = sqrt(P_est[i, 2, 2])

e_fig = plt.figure()
ax = e_fig.add_subplot(111)
ax.plot(t[:], (std_x[:]), label='Sigma X')
ax.plot(t[:], (std_y[:]), label='Sigma Y')
ax.plot(t[:], (std_theta[:]), label='Sigma Theta')
ax.set_xlabel('Time [s]')
ax.set_ylabel('theta [rad]')
ax.set_title('Estimated trajectory')
ax.legend()
plt.savefig('images/week2_uncertainity_no_update_step.png', dpt=300)
plt.show()

with open('submission.pkl', 'wb') as f:
    pickle.dump(x_est, f, pickle.HIGHEST_PROTOCOL)