import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Grid setup
Ni = 25   # grid points per dimension
Nj = 100
L = 1
Lx = Ni * L
Ly = Nj * L
dx = Lx / (Ni - 1)
dy = Ly / (Nj - 1)

# Material properties (example: aluminum)
alpha = 9.7e-5   # thermal diffusivity (m^2/s)
k = 200.0        # thermal conductivity (W/mK)
h = 10.0         # convection coefficient (W/m²K)
T_inf = 15.0     # ambient air temperature (°C)

# Time step from stability criterion
dt = 0.25 * dx**2 / alpha
steps = 200


# Initialize temperature field
hot_point = 30
T = np.ones((Ni, Nj)) * 20.0
T[Ni//2, :] = hot_point
#
# for x in range(Nj):
#     T_val = x**2 + x*2
#     T[Ni//2, x] = T_val/100
# print(T[Ni//2, :])

# Update function
def step(T, t, dt):
    T_new = T.copy()
    for i in range(1, Ni-1):
        for j in range(1, Nj-1):
            T_new[i, j] = T[i, j] + alpha*dt*(
                (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / dx**2 +
                (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / dy**2
            )

    # Left wall: constant hot temperature
    # T_new[:, 0] = 500.0
    T_new[Ni//2, :] = hot_point

    # for x in range(Nj):
    #     T_val = x ** 2 + x * 2
    #     T_new[Ni // 2, x] = T_val / 100

    # Right wall: convection to air (Robin BC)

    Bi = h*dx/k   # finite-difference Biot number

    # T_new[:, -1] = (T[:, -2] + Bi * T_inf) / (1 + Bi)
    # T_new[:, 0] = (T[:,1] + Bi * T_inf) / (1 + Bi)
    T_new[0, :] = (T[1, :] + Bi * T_inf) / (1 + Bi)
    T_new[-1, :] = (T[-2, :] + Bi * T_inf) / (1 + Bi)

    # Top and bottom: insulated (zero flux)
    # Acts like infinite region

    # T_new[0, :] = T_new[1, :]
    # T_new[-1, :] = T_new[-2, :]
    T_new[:,-1] = T_new[:,-2]
    T_new[:,0] = T_new[:,1]

    return T_new



avg_temp = np.mean(T)
avg_temp_prev = np.mean(T)
avg_diff = 1
t = 0

while avg_diff > 0.005:
    t += dt
    T_new = step(T, t, dt)

    avg_temp = np.mean(T_new)
    avg_diff = np.abs(avg_temp - avg_temp_prev) / avg_temp_prev
    avg_temp_prev = avg_temp

print(f"Time required for 2% difference: {t:.2f}s")

