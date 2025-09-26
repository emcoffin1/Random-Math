from scipy import integrate, linalg
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Initial State Variable
X_0 = np.array([-9000, 9000, 8, 1])
m = 500


# Earth State Variable and stats
E = np.array([0, 0])
E_R = 6356      # km
mu = 398600.4418


# Function
def func(t, state) -> np.ndarray:
    rx, ry, vx, vy = state

    r = np.array([rx-E[0], ry-E[1]])
    r_n = linalg.norm(r)
    a = -mu * r / r_n**3

    return np.array([vx, vy, a[0], a[1]])

# Impact information
def impact_event(t, state):
    rx, ry, vx, vy = state
    r = np.linalg.norm([rx, ry])
    dist = r - E_R
    return dist

impact_event.terminal = True
impact_event.direction = -1

# Integration time span
t_span = (0, 12*3600)
t_eval = np.linspace(t_span[0], t_span[1], 5000)

# Integrate
sol = solve_ivp(func, t_span, X_0, t_eval=t_eval, rtol=1e-9, atol=1e-12, events=impact_event)
# Trajectory
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
# plt.legend()
plt.show()




