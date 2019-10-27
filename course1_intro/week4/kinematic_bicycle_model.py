import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import *

class Bicycle():
    def __init__(self):
        self.xc = 0
        self.yc = 0
        self.theta = 0
        self.delta = 0
        self.beta = 0
        
        self.L = 2
        self.lr = 1.2
        self.w_max = 1.22
        
        self.sample_time = 0.01
        
    def reset(self):
        self.xc = 0
        self.yc = 0
        self.theta = 0
        self.delta = 0
        self.beta = 0

class Bicycle(Bicycle):
    def step(self, v, w):
        
        if (w > self.w_max):
            w = self.w_max
        elif (w < -self.w_max):
            w = -self.w_max
        
        # From the equations presented in the slides, center of gravity equations
        xc_dot = v * np.cos(self.theta + self.beta)
        yc_dot = v * np.sin(self.theta + self.beta)
        theta_dot = v * np.cos(self.beta) * np.tan(self.delta) / self.L
        self.beta = np.arctan(self.lr *np.tan(self.delta) / self.L)
        # Update the states 
        self.xc += xc_dot * self.sample_time
        self.yc += yc_dot * self.sample_time
        self.theta += theta_dot * self.sample_time
        self.delta += w * self.sample_time

sample_time = 0.01
time_end = 20
model = Bicycle()
model_beta0 = Bicycle()

# set delta directly
model.delta = np.arctan(2/10)
model_beta0.delta = np.arctan(2/10)

t_data = np.arange(0,time_end,sample_time)
# Initialize all data to zeros, at the same size of t_data
x_data = np.zeros_like(t_data)
y_data = np.zeros_like(t_data)
x_beta0 = np.zeros_like(t_data)
y_beta0 = np.zeros_like(t_data)

for i in range(t_data.shape[0]):
    x_data[i] = model.xc
    y_data[i] = model.yc
    model.step(np.pi, 0)

    x_beta0[i] = model_beta0.xc
    y_beta0[i] = model_beta0.yc
    model_beta0.step(np.pi, 0)
    
    # model.beta = 0
    model_beta0.beta = 0

plt.axis('equal')
plt.plot(x_data, y_data,label='Learner Model')
plt.legend()
plt.savefig('week4/images/initial_bicycle_solution.png', dpt=300)
plt.show()

radius_computed = np.zeros_like(x_data)
for i in range(x_data.shape[0]):
    radius_computed[i] = sqrt((x_data[i] - 0)**2 + (y_data[i] - 10)**2)

plt.axis('equal')
plt.plot(t_data, radius_computed,label='Learner Model')
plt.legend()
plt.savefig('week4/images/initial_bicylce_radius.png', dpt=300)
plt.show()

## Compare the results when beta is set to 0
plt.axis('equal')
plt.plot(x_data, y_data,label='Learner Model')
plt.plot(x_beta0, y_beta0,label='Beta set to 0')
plt.legend()
plt.savefig('week4/images/compare_beta_0_trajectory.png', dpt=300)
plt.show()
# Compute radius of beta == 0
radius_computed_beta0 = np.zeros_like(x_data)
for i in range(x_data.shape[0]):
    radius_computed_beta0[i] = sqrt((x_beta0[i] - 0)**2 + (y_beta0[i] - 10)**2)

plt.axis('equal')
plt.plot(t_data, radius_computed,label='Learner Model')
plt.plot(t_data, radius_computed_beta0,label='Learner Model')
plt.legend()
plt.savefig('week4/images/compare_beta_0_radius.png', dpt=300)
plt.show()

sample_time = 0.01
time_end = 20
model.reset()

### NOTE ####
# After reset() delta is now 0 and not being set manually

t_data = np.arange(0,time_end,sample_time)
x_data = np.zeros_like(t_data)
y_data = np.zeros_like(t_data)
delta_data = np.zeros_like(t_data)

for i in range(t_data.shape[0]):
    x_data[i] = model.xc
    y_data[i] = model.yc
    delta_data[i] = model.delta
    if model.delta < np.arctan(2/10):
        model.step(np.pi, model.w_max)
    else:
        model.step(np.pi, 0)

plt.axis('equal')
plt.plot(x_data, y_data,label='Learner Model')
plt.savefig('week4/images/trajectory_passing_omega.png', dpt=300)
plt.legend()
plt.show()

radius_computed = np.zeros_like(x_data)
for i in range(x_data.shape[0]):
    radius_computed[i] = sqrt((x_data[i] - 0)**2 + (y_data[i] - 10)**2)

plt.axis('equal')
plt.plot(t_data, radius_computed,label='Learner Model')
plt.legend()
plt.savefig('week4/images/passing_omega_radius.png', dpt=300)
plt.show()

plt.axis('equal')
plt.plot(t_data, delta_data,label='Learner Model')
plt.legend()
plt.savefig('week4/images/passing_omega_delta.png', dpt=300)
plt.show()

sample_time = 0.01
time_end = 30
model.reset()

t_data = np.arange(0,time_end,sample_time)
x_data = np.zeros_like(t_data)
y_data = np.zeros_like(t_data)
v_data = np.zeros_like(t_data)
w_data = np.zeros_like(t_data)
delta_data = np.zeros_like(t_data)

# ==================================
#  Learner solution begins here
# ==================================
radius = 8
# This is the max steering angle
max_angle_percentage = 0.95
maximum_delta = max_angle_percentage * np.arctan(model.L / radius)  
# Using the equation provided to calculate velocity v = d/t
# which goes to (2 * pi * radius) / time_end which for the 
# previous examples worked out to (2 * pi * 10) / 20, or just pi
# now we have two circles for the total distance so we have
# v = 2 * (2 * pi * 8) / 30
v_data[:] = 2 * (2 * np.pi * radius) / (time_end)

w_sign = np.zeros_like(v_data)
# How long is each quarter
quarter_time = int(t_data.shape[0] / 8)

# The first quarter is counter clockwise so negative omega
w_sign[0:quarter_time] = 1
w_sign[quarter_time+1:quarter_time*5] = -1
w_sign[quarter_time*5+1:] = 1
plt.plot(t_data, w_sign, label='Sign of Steering Angle Rate')
plt.savefig('week4/images/figure8_sign_of_omega.png')
plt.show()

for i in range(t_data.shape[0]):
    x_data[i] = model.xc
    y_data[i] = model.yc
    delta_data[i] = model.delta
    # Calculate the steering angle rate, make sure to include the sign
    w = model.w_max * w_sign[i]
    # Make sure the model has not exceeded max steering angle
    if w_sign[i] < 0 and model.delta <= -maximum_delta:
        # We have the max steering angle, so hold that angle by setting
        # angle rate to 0
        w = 0
    elif w_sign[i] > 0 and model.delta >= maximum_delta:
        w = 0
    
    model.step(v_data[i], w)
    w_data[i] = w


plt.axis('equal')
plt.grid()
plt.plot(x_data, y_data)
plt.savefig('week4/images/figure8_trajectory.png')
plt.show()

plt.grid()
plt.plot(t_data, w_data)
plt.savefig('week4/images/figure8_omega.png')
plt.show()

plt.grid()
plt.plot(t_data, delta_data)
plt.savefig('week4/images/figure8_delta.png')
plt.show()