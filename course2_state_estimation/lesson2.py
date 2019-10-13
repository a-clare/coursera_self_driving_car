import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

I = np.array([0.2, 0.3, 0.4, 0.5, 0.6])
V = np.array([1.23, 1.38, 2.06, 2.47, 3.17])

## Batch Solution
H = np.ones((5,2))
H[:, 0] = I
x_ls = inv(H.T.dot(H)).dot(H.T.dot(V))
print('The parameters of the line fit are ([R, b]):')
print(x_ls)

#Plot
I_line = np.arange(0, 0.8, 0.1)
V_line = x_ls[0]*I_line + x_ls[1]

plt.scatter(I, V)
plt.plot(I_line, V_line)
plt.xlabel('current (A)')
plt.ylabel('voltage (V)')
plt.grid(True)
plt.show()


# State vector. Where the unknown states are R, and b
x_k = np.array([4.0, 0.0])
# State uncertainity matrix
P_k = np.array([[10.0, 0], [0.0, 0.2]])
# Assuming all measurements are equally weighted. This matrix is the measurement
# noise matrix, not to be confused with the R for resistance. Keeping the 
# notation the same as in the formula
R = 0.0225


num_meas = I.shape[0]
x_hist = np.zeros((num_meas + 1, 2))
P_hist = np.zeros((num_meas + 1, 2, 2))

x_hist[0] = x_k
P_hist[0] = P_k

for k in range(num_meas):

    # The jacobian matrix.
    # 1 row since we have 1 observation, and 2 columns since there
    # are 2 unknowns, R (resistance) and b
    H_k = np.array([I[0], 1])

    # Create a temporary matrix to hold the P * H^T calculation.
    # This matrix is used twice in the calculation of K
    temp = np.dot(P_k, H_k.T)
    temp2 = np.dot(H_k, temp)
    # Normally this would have to be a matrix inverse but we are getting a 
    # a floating point value (or a 1x1 matrix), so just 1.0 / value to
    # get inverse
    temp2 = 1.0 / (temp2 + R)
    K = np.dot(temp, temp2)
    # Innovation sequence
    z = V[0] - np.dot(H_k, x_k.T)
    x_k = x_k + np.dot(K, z)
    P_k = np.dot(np.eye(2, 2) - np.dot(K, H_k), P_k)
    P_hist[k+1] = P_k
    x_hist[k+1] = x_k

print('The parameters of the line fit are ([R, b]):')
print(x_k)

#Plot
plt.scatter(I, V, label='Data')
plt.plot(I_line, V_line, label='Batch Solution')
plt.xlabel('current (A)')
plt.ylabel('voltage (V)')
plt.grid(True)

I_line = np.arange(0, 0.8, 0.1)
for k in range(num_meas):
    V_line = x_hist[k,0]*I_line + x_hist[k,1]
    plt.plot(I_line, V_line, label='Measurement {}'.format(k))

plt.legend()
plt.show()