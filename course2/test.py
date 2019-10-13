import numpy as np
import math
import random
import matplotlib.pyplot as plt

x = np.array([
    [1.0, 10.0],
    [2.0, 20.0],
    [3.0, 30.0],
    [4.0, 40.0],
    [5.0, 50.0]]).reshape(5, 2)

x0 = np.mean(x, axis=0)

var = np.zeros((2, 2))
for i in range(5):
    t = x[i, :].reshape(2, 1) - x0.reshape(2, 1)
    var = var +  np.dot(t, t.transpose())
