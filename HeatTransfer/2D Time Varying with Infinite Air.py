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
hot_point = 40
T = np.ones((Ni, Nj)) * 80.0
T[Ni//2:(Ni//2)+2, :] = hot_point
T[Ni//2:(Ni//2)-2, :] = hot_point

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
    # T_new[Ni//2, :] = hot_point

    T_new[Ni//2:(Ni//2)+5, :] = hot_point
    T_new[Ni//2:(Ni//2)-5, :] = hot_point

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

# Probe point (center of the plate)
i_probe, j_probe = Ni//2, Nj//2
T_probe = []
time_history = []
T_avg = []

# Setup figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left plot: probe temperature history
line1, = ax1.plot([], [], label="Center Temp")
line2, = ax1.plot([], [], label="Average Temp")
ax1.set_xlim(0, steps*dt)
ax1.set_ylim(0, 110)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Temperature (°C)")
ax1.legend()
ax1.set_title("Probe Temperature vs Time")

# Right plot: animated temperature field
im = ax2.imshow(T, cmap='hot', origin='lower', vmin=0, vmax=100)
fig.colorbar(im, ax=ax2, label="Temperature (°C)")

# Animation update
def animate(frame):
    global T
    t = frame * dt
    T = step(T, t, dt)
    print(t)
    # Update probe history
    T_probe.append(T[i_probe, j_probe])
    time_history.append(t)
    T_avg.append(np.mean(T))

    line1.set_data(time_history, T_probe)
    ax1.set_xlim(0, max(time_history)+dt)

    line2.set_data(time_history, T_avg)

    # Update temperature field
    im.set_data(T)
    # ax2.set_title(f"Step {frame}, Time = {t:.2f}s")

    return [line1, line2, im]

anim = FuncAnimation(fig, animate, frames=steps, interval=50, blit=False)
plt.tight_layout()
plt.show()
