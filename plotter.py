import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0, 3600, 1000)
T = 100.4 * np.exp(-5.122e-5*t)

plt.plot(t, T)
plt.show()