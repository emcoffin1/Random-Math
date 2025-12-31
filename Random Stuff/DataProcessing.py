import numpy as np
import matplotlib.pyplot as plt

x,y = np.loadtxt("sin_noisy.txt", delimiter=',', skiprows=1, unpack=True)

dydx = np.gradient(x,y)

filt = np.ones(15)/15

y_smooth = np.convolve(y, filt, mode='valid')
dysdx = np.gradient(y_smooth, x[7:-7])

plt.plot(x,y, "--", linewidth=1)
plt.plot(x[7:-7],y_smooth, "--", linewidth=1)
plt.plot(x, dydx, "--", linewidth=1)
plt.plot(x[7:-7], dysdx, "--", linewidth=1)
plt.show()