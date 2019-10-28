import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
plt.show()

time_end = 20
t_data = np.arange(0,time_end,sample_time)
x_data = np.zeros_like(t_data)

# reset the states
model.reset()

# ==================================
#  Learner solution begins here
# ==================================

# ==================================
#  Learner solution ends here
# ==================================

# Plot x vs t for visualization
plt.plot(t_data, x_data)
plt.show()