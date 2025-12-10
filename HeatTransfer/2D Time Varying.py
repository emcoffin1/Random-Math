import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Grid setup
N = 50   # grid points per dimension
L = 1.0
dx = L / (N - 1)
dy = dx

# Material properties (example: aluminum)
alpha = 9.7e-5   # thermal diffusivity (m^2/s)

# Time step from stability criterion
dt = 0.25 * dx**2 / alpha
steps = 1000

# Initialize temperature field
T = np.ones((N, N)) * 20.0
T[:, 0] = 100.0   # left edge hot (initial condition)
T[:, -1] = 0.0    # right edge cold

# Update function
def step(T, t, dt):
    T_new = T.copy()
    for i in range(1, N-1):
        for j in range(1, N-1):
            T_new[i, j] = T[i, j] + alpha*dt*(
                (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / dx**2 +
                (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / dy**2
            )

    # Sinusoidal boundary condition on the left wall
    T_mean = 50.0
    A = 50.0
    f = 0.01  # Hz
    T_left = T_mean + A * np.sin(2*np.pi*f*t)

    # Boundary conditions
    T_new[:, 0] = T_left        # left wall = sinusoidal
    T_new[:, -1] = 0.0          # right wall = fixed cold
    T_new[0, :] = T_new[1, :]   # top insulated
    T_new[-1, :] = T_new[-2, :] # bottom insulated
    return T_new

# Probe point (center of the plate)
i_probe, j_probe = N//2, N//2
T_probe = []
time_history = []

# Setup figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Left plot: probe temperature history
line1, = ax1.plot([], [], label="Center Temp")
line2, = ax1.plot([], [], 'r--', label="Boundary Temp")
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

    # Update probe history
    T_probe.append(T[i_probe, j_probe])
    time_history.append(t)

    # Update left plot
    line1.set_data(time_history, T_probe)
    line2.set_data(time_history, 50 + 50*np.sin(2*np.pi*0.01*np.array(time_history)))

    ax1.set_xlim(0, max(time_history)+dt)

    # Update right plot
    im.set_data(T)
    ax2.set_title(f"Step {frame}, Time = {t:.2f}s")

    return [line1, line2, im]

anim = FuncAnimation(fig, animate, frames=steps, interval=50, blit=False)
plt.tight_layout()
plt.show()
