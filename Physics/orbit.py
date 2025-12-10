import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants
mu = 398600.4418  # Earth's gravitational parameter [km^3/s^2]

# Two-body dynamics
def two_body(t, state):
    x, y, vx, vy = state
    r = np.sqrt(x**2 + y**2)
    ax = -mu * x / r**3
    ay = -mu * y / r**3
    return [vx, vy, ax, ay]

# Initial conditions: circular orbit at ~7000 km radius
r0 = 7000.0  # km
v0 = np.sqrt(mu / r0)  # circular velocity
state0 = [r0, 0, 0, v0]  # [x0, y0, vx0, vy0]

# Time span (1.5 orbital periods)
T = 2 * np.pi * np.sqrt(r0**3 / mu)  # orbital period
t_span = (0, 1.5 * T)
t_eval = np.linspace(0, 1.5 * T, 1000)

# Integrate
sol = solve_ivp(two_body, t_span, state0, t_eval=t_eval, rtol=1e-9, atol=1e-12)

# Extract solution
x, y = sol.y[0], sol.y[1]

# Plot orbit
plt.figure(figsize=(6,6))
plt.plot(x, y, label="Orbit")
plt.plot(0, 0, "ro", label="Earth")
plt.xlabel("x [km]")
plt.ylabel("y [km]")
plt.gca().set_aspect("equal", adjustable="box")
plt.legend()
plt.title("Two-Body Orbit with solve_ivp")
plt.show()
