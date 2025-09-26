import numpy as np
import matplotlib.pyplot as plt

# 2D Square plate that is 1x1m

# Grid setup
N = 10
T = np.zeros((N, N))

# Boundary Conditions
T[:, 0]  = 100   # left edge hot
T[:, -1] = 0     # right edge cold

# Jacobi solver
tol = 1e-4
error = 1

while error > tol:
    T_old = T.copy()

    for i in range(1, N-1):
        for j in range(1, N-1):
            T[i, j] = 0.25 * (T_old[i+1, j] + T_old[i-1, j] + T_old[i, j+1] + T_old[i, j-1])

    T[0, :]  = T[1, :]
    T[-1, :] = T[-2, :]

    error = np.max(np.abs(T - T_old))

print(T)

plt.imshow(T, cmap='hot', origin='lower')
plt.colorbar(label="Temp (C)")
plt.show()
