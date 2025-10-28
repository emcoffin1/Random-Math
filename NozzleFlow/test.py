import numpy as np

r = 0.15/2
t = 0.002
d = 0.005

N = 2*np.pi*(r+t+d)/(1.3*d)
print(N)