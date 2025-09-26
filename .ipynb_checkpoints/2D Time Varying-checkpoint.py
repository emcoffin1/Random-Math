import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Grid setup
N = 50   # grid points per dimension
L = 1.0
dx = L / (N - 1)
dy = dx

# Material properties (example: aluminum)
alpha = 9.7e-5   # thermal diffusivity (m^2/s)

# Time step from stability criterion
dt = 0.25 * dx**2 / alpha
steps = 200

# Initialize temperature field
T = np.ones((N, N)) * 20.0
T[:, 0] = 100.0   # left edge hot
T[:, -1] = 0.0    # right edge cold

# Update function
def step(T):
    T_new = T.copy()
    for i in range(1, N-1):
        for j in range(1, N-1):
            T_new[i, j] = T[i, j] + alpha*dt*(
                (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / dx**2 +
                (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / dy**2
            )
    # Boundary conditions
    T_new[:, 0] = 100.0
    T_new[:, -1] = 0.0
    T_new[0, :] = T_new[1, :]       # top insulated
    T_new[-1, :] = T_new[-2, :]     # bottom insulated
    return T_new

# Plot setup
fig, ax = plt.subplots()
im = ax.imshow(T, cmap='hot', origin='lower', vmin=0, vmax=100)
fig.colorbar(im, ax=ax, label="Temperature (°C)")

def animate(frame):
    global T
    T = step(T)
    im.set_data(T)
    ax.set_title(f"Step {frame}")
    return [im]

anim = FuncAnimation(fig, animate, frames=steps, interval=50, blit=True)

# Display in notebook
HTML(anim.to_jshtml())
