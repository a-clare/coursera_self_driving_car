import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from numpy import genfromtxt
import math

save_path = '/home/adam/projects/coursera-self-driving-car/course1_intro/week7/images/'
way_point_file = "/home/adam/projects/coursera-self-driving-car/course1_intro/carla/CarlaSimulator/PythonClient/Course1FinalProject/racetrack_waypoints.txt"

waypoints = genfromtxt(way_point_file, delimiter=',')

plt.plot(waypoints[:, 0], waypoints[:, 1])
plt.plot(waypoints[0, 0], waypoints[0, 1], 'r*')
plt.grid()
plt.savefig(save_path + 'x_y.png')
plt.show()

plt.plot(waypoints[:, 2])
plt.grid()
plt.savefig(save_path + 'v.png')
plt.show()

dt = 0.033
