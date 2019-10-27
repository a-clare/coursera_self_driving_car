import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
model_beta_set_to0 = Bicycle()

# set delta directly
model.delta = np.arctan(2/10)
model_beta_set_to0.delta = np.arctan(2/10)

t_data = np.arange(0,time_end,sample_time)
# Initialize all data to zeros, at the same size of t_data
x_data = np.zeros_like(t_data)
y_data = np.zeros_like(t_data)
x_data2 = np.zeros_like(t_data)
y_data2 = np.zeros_like(t_data)

for i in range(t_data.shape[0]):
    x_data[i] = model.xc
    y_data[i] = model.yc
    model.step(np.pi, 0)

    x_data2[i] = model_beta_set_to0.xc
    y_data2[i] = model_beta_set_to0.yc
    model_beta_set_to0.step(np.pi, 0)
    
    # model.beta = 0
    model_beta_set_to0.beta = 0
    
plt.axis('equal')
plt.plot(x_data, y_data,label='Learner Model')
plt.plot(x_data2, y_data2s,label='Beta set to 0')
plt.legend()
plt.show()