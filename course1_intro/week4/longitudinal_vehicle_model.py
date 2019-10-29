import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import *

class Vehicle():
    def __init__(self):
 
        # ==================================
        #  Parameters
        # ==================================
    
        #Throttle to engine torque
        self.a_0 = 400
        self.a_1 = 0.1
        self.a_2 = -0.0002
        
        # Gear ratio, effective radius, mass + inertia
        self.GR = 0.35
        self.r_e = 0.3
        self.J_e = 10
        self.m = 2000
        self.g = 9.81
        
        # Aerodynamic and friction coefficients
        self.c_a = 1.36
        self.c_r1 = 0.01
        
        # Tire force 
        self.c = 10000
        self.F_max = 10000
        
        # State variables
        self.x = 0
        self.v = 5
        self.a = 0
        self.w_e = 100
        self.w_e_dot = 0
        
        self.sample_time = 0.01
        
    def reset(self):
        # reset state variables
        self.x = 0
        self.v = 5
        self.a = 0
        self.w_e = 100
        self.w_e_dot = 0

    def step(self, throttle, alpha):
        
        # Compute engine torque
        Te = throttle * (self.a_0 + self.a_1 * self.w_e + self.a_2 * self.w_e**2)
        # Aerodynamic drag
        Fe = self.c_a * self.v**2
        # Rolling friction
        Rx = self.c_r1 * self.v
        # Gravitational force
        Fg = self.m * self.g * np.sin(alpha)
        Fload = Fe + Rx + Fg
        # Torque 
        self.w_e_dot = (Te - self.GR * self.r_e * Fload) / self.J_e 
        # Wheel angular velocity
        omega_w = self.GR * self.w_e
        # Slip ratio
        s = (omega_w * self.r_e - self.v) / self.v
        # Tire force, need to check conditions according to notebook
        Fx = self.F_max
        if abs(s) < 1:
            Fx = self.c * s
        
        # Calculate acceleration from m*a = Fx - Fload
        self.a = (Fx - Fload) / self.m
        
        # Update state. These arent explicity given and had to be worked out
        self.w_e += self.w_e_dot * self.sample_time
        self.v += self.a * self.sample_time
        self.x += (self.v * self.sample_time) - (0.5 * self.a * self.sample_time**2)
        
sample_time = 0.01
time_end = 100
model = Vehicle()

t_data = np.arange(0,time_end,sample_time)
v_data = np.zeros_like(t_data)

# throttle percentage between 0 and 1
throttle = 0.2

# incline angle (in radians)
alpha = 0

for i in range(t_data.shape[0]):
    v_data[i] = model.v
    model.step(throttle, alpha)
    
plt.plot(t_data, v_data)
plt.savefig('week4/images/longitudinal_constant_throttle_flat_ground.png', dpt=300)
plt.show()

time_end = 20
t_data = np.arange(0,time_end,sample_time)
x_data = np.zeros_like(t_data)

# reset the states
model.reset()
# Keep a record of the throttle commands to make sure it matches figure in notebook
all_throttle = np.zeros_like(t_data)
# Keep a record of the alpha commands to make sure it matches figure in notebook
all_alpha = np.zeros_like(t_data)
for i in range(t_data.shape[0]):
    # If we are between 0 and 5 seconds, throttle increase
    if (t_data[i] < 5):
        # 5.0 comes from 5 second duration
        # In the notebook assignment it says the throttle starts at 0.2
        throttle = 0.2 + (0.5 - 0.2) / 5.0 * t_data[i]
    # If we are between 5 and 15 seconds constant throttle at 50%
    elif (t_data[i] < 15):
        throttle = 0.5
    # If we are between 15 and 20 seconds, decrease throttle to 0
    else:
        # 5.0 comes from 5 second duration
        # In the notebook assignment it says the throttle starts at 0.2
        throttle = (0.0 - 0.5) / 5.0 * (t_data[i] - 20.0)
    
    if (model.x < 60):
        alpha = atan(3.0/60.0)
    elif (model.x < 150):
        alpha = atan(9.0 / 90.0)
    else:
        alpha = 0
        
    model.step(throttle, alpha)
    all_throttle[i] = throttle
    all_alpha[i] = alpha
    x_data[i] = model.x
    
# Plot x vs t for visualization
plt.plot(t_data, x_data)
plt.grid()
plt.savefig('week4/images/longitudinal_x_position.png', dpt=300)
plt.show()

plt.plot(t_data, all_throttle)
plt.grid()
plt.savefig('week4/images/longitudinal_throttle_ramp.png', dpt=300)
plt.show()

plt.plot(t_data, all_alpha)
plt.grid()
plt.savefig('week4/images/longitudinal_alpha_ramp.png', dpt=300)
plt.show()