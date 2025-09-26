import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Initial state: [x0, y0, vx0, vy0]
X_0 = np.array([-9000, 9000, 3, -5])  # km, km/s
# tweak velocity values to get a flyby/ricochet effect

# Earth parameters
E = np.array([0, 0])     # Earth at origin
E_R = 6356               # Earth radius [km]
mu = 398600.4418         # Earth's mu [km^3/s^2]

# Dynamics function
def func(t, state) -> np.ndarray:
    rx, ry, vx, vy = state
    r_vec = np.array([rx - E[0], ry - E[1]])
    r = np.linalg.norm(r_vec)
    a = -mu * r_vec / r**3
    return [vx, vy, a[0], a[1]]

# Integration time span
t_span = (0, 12 * 3600)   # 12 hours, in seconds
t_eval = np.linspace(t_span[0], t_span[1], 5000)

# Integrate
sol = solve_ivp(func, t_span, X_0, t_eval=t_eval, rtol=1e-9, atol=1e-12)

# Extract trajectory
x, y = sol.y[0], sol.y[1]

# Plot
plt.figure(figsize=(7,7))
plt.plot(x, y, label="Asteroid trajectory")
circle = plt.Circle((0,0), E_R, color='b', alpha=0.3, label="Earth")
plt.gca().add_artist(circle)
plt.plot(0, 0, "bo", label="Earth center")
plt.xlabel("x [km]")
plt.ylabel("y [km]")
plt.gca().set_aspect("equal", adjustable="box")
plt.title("Asteroid Flyby around Earth")
plt.legend()
plt.show()
