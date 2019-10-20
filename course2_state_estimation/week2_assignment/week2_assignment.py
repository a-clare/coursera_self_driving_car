import pickle
import numpy as np
import matplotlib.pyplot as plt
from math import *

with open('week2_assignment/data.pickle', 'rb') as f:
    data = pickle.load(f)

t = data['t']  # timestamps [s]

x_init  = data['x_init'] # initial x position [m]
y_init  = data['y_init'] # initial y position [m]
th_init = data['th_init'] # initial theta position [rad]
fig_name = '_om_to_0.1_meas_noise'
# input signal
v  = data['v']  # translational velocity input [m/s]
om = data['om']  # rotational velocity input [rad/s]
v_est = np.zeros((len(v)))
# bearing and range measurements, LIDAR constants
b = data['b']  # bearing to each landmarks center in the frame attached to the laser [rad]
r = data['r']  # range measurements [m]
l = data['l']  # x,y positions of landmarks [m]
d = data['d']  # distance between robot center and laser rangefinder [m]

v_var = 0.02 # translation velocity variance  
om_var = 0.008  # rotational velocity variance 
print(np.var(om))
print(np.var(v))
r_var = 0.01  # range measurements variance
b_var = 0.005  # bearing measurement variance

Q_km = np.diag([v_var, om_var]) # input noise covariance 
cov_y = np.diag([r_var, b_var])  # measurement noise covariance 

x_est = np.zeros([len(v), 3])  # estimated states, x, y, and theta
P_est = np.zeros([len(v), 3, 3])  # state covariance matrices

x_est[0] = np.array([x_init, y_init, th_init]) # initial state
P_est[0] = np.diag([1, 1, 0.1]) # initial state covariance
# Initialze the state uncertainity matrix
P_check = P_est[0]
x_check = x_est[0].reshape(3, 1)

def wraptopi(x):
    if x > np.pi:
        x = x - (np.floor(x / (2 * np.pi)) + 1) * 2 * np.pi
    elif x < -np.pi:
        x = x + (np.floor(x / (-2 * np.pi)) + 1) * 2 * np.pi
    return x

# lk = landmark x,y
# rk = range measurements
# bk = bearing to each landmark
def measurement_update(lk, rk, bk, P_check, x_check):
    
    # 1. Compute measurement Jacobian
    # Our measurement model consists of r and phi. Distance and 
    # bearing to a landmark. Need to take the partial derivatives 
    # of the two equations (shown in the homework, y_k) with respect
    # to our three unkowns x_k, y_k, omega_k 
    # Our first measurement equation is:
    # r = sqrt((x_l - x_k - d*cos(theta_k)) ^2 + (y_l - y_k - d*sin(theta_k)) ^2)
    cos_theta = cos(x_check[2])
    sin_theta = sin(x_check[2])
    # These are the two terms inside the square root
    delta_x = lk[0] - x_check[0] - d * cos_theta
    delta_y = lk[1] - x_check[1] - d * sin_theta
    r = sqrt(delta_x**2 + delta_y**2)
    # The second measurement model
    phi = wraptopi(atan2(delta_y, delta_x) - x_check[2])
    
    # Our jacobian is a 2x3, since we have 2 observations (r/phi), and 
    # 3 unknowns (x, y, theta)
    H = np.zeros((2, 3))
    H[0, 0] = -delta_x / r
    H[0, 1] = -delta_y / r
    H[0, 2] = d * (delta_x * sin_theta - delta_y * cos_theta) / r
    H[1, 0] = delta_y / r**2
    H[1, 1] = -delta_x / r**2
    H[1, 2] = -d * (delta_y * sin_theta + delta_x * cos_theta) / r**2
    
    M = np.eye(2)
    # 2. Compute Kalman Gain
    K = P_check @ H.T @ np.linalg.inv(H @ P_check @ H.T + M @ cov_y @ M.T)
    # Innovation sequence
    z = np.vstack([rk - r, wraptopi(bk) - phi])
    
    # 3. Correct predicted state (remember to wrap the angles to [-pi,pi])
    x_check = x_check + K @ z
    x_check[2] = wraptopi(x_check[2])

    # 4. Correct covariance
    P_check = (np.eye(3) - K @ H) @ P_check
    return x_check, P_check


for k in range(1, len(t)):  # start at 1 because we've set the initial prediciton

    delta_t = t[k] - t[k - 1]  # time step (difference between timestamps)
    # 1. Update state with odometry readings (remember to wrap the angles to [-pi,pi])
    # Break out theta before update since its used in several spots
    theta = x_check[2]
    sin_theta_delta_t = sin(theta) * delta_t
    cos_theta_delta_t = cos(theta) * delta_t
    # Propogate the state forward to the current step
    x_check[0] += cos_theta_delta_t * v[k-1]
    x_check[1] += sin_theta_delta_t * v[k-1]
    x_check[2] += delta_t * om[k]
    x_check[2] = wraptopi(x_check[2])
    
    # 2. Motion model jacobian with respect to last state
    F_km = np.array([
        [1.0, 0.0, -sin_theta_delta_t * v[k-1]],
        [0.0, 1.0, cos_theta_delta_t * v[k-1]],
        [0.0, 0.0, 1.0]])

    # 3. Motion model jacobian with respect to noise
    L_km = np.array([
        [cos_theta_delta_t, 0.0],
        [sin_theta_delta_t, 0.0],
        [0.0, delta_t]])
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
    v_est[k] = sqrt((x_est[k, 0] - x_est[k-1, 0]) ** 2 + (x_est[k, 1] - x_est[k-1, 1]) ** 2)


e_fig = plt.figure()
ax = e_fig.add_subplot(111)
ax.plot(t[:], v[:], 'r--')
ax.plot(t[:], v_est[:])
ax.set_xlabel('t [s]')
ax.set_ylabel('v [m/s]')
ax.set_title('Velocity')
plt.savefig('images/week2_v' + fig_name + '.png', dpt=300)
plt.show()

e_fig = plt.figure()
ax = e_fig.add_subplot(111)
ax.plot(t[:], om[:])
ax.set_xlabel('t [s]')
ax.set_ylabel('omega [m/s]')
ax.set_title('Velocity - omega')
plt.savefig('images/week2_omega' + fig_name + '.png', dpt=300)
plt.show()

e_fig = plt.figure()
ax = e_fig.add_subplot(111)
ax.plot(x_est[:, 0], x_est[:, 1])
plt.plot(x_init, y_init, 'r*')
plt.plot(x_est[-1, 0], x_est[-1, 1], 'r^')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Estimated trajectory')
plt.savefig('images/week2_trajectory_xy_' + fig_name + '.png', dpt=300)
plt.show()

e_fig = plt.figure()
ax = e_fig.add_subplot(111)
ax.plot(t[:], x_est[:, 2])
ax.set_xlabel('Time [s]')
ax.set_ylabel('theta [rad]')
ax.set_title('Estimated trajectory')
plt.savefig('images/week2_trajectory_theta_' + fig_name + '.png', dpt=300)
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
plt.savefig('images/week2_uncertainity_' + fig_name + '.png', dpt=300)
plt.show()

with open('week2_assignment/submission.pkl', 'wb') as f:
    pickle.dump(x_est, f, pickle.HIGHEST_PROTOCOL)