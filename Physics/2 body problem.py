from scipy import integrate, linalg
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


m = 5e11
G = 6.674e-11

# State Variable
# rx1, ry1, vx1, vy1, etc
X = np.array([-5, 0, 1,
              0, 5, 0,
              5, 0, 1,
              0.5, 3.5, 2.5])


def func(t, state):
    x1, y1, z1, vx1, vy1, vz1, x2, y2, z2, vx2, vy2, vz2 = state
    r1 = np.array([x1, y1, z1])
    r2 = np.array([x2, y2, z2])

    r_vec = linalg.norm(r1-r2)

    a1 = G * m * (r2-r1) / r_vec**3
    a2 = G * m * (r1-r2) / r_vec**3

    return [vx1, vy1, vz1, a1[0], a1[1], a1[2],
            vx2, vy2, vz2, a2[0], a2[1], a2[2]]

# Integration time span
t_span = (0, 500)
t_eval = np.linspace(t_span[0], t_span[1], 5000)

# Integrate
sol = integrate.solve_ivp(func, t_span, X, t_eval=t_eval, rtol=1e-9, atol=1e-12)
# Trajectory
px1 = sol.y[0]
py1 = sol.y[1]
pz1 = sol.y[2]
px2 = sol.y[6]
py2 = sol.y[7]
pz2 = sol.y[8]

# plt.figure(figsize=(7,7))
# plt.plot(px1, py1, label="Particle 1")
# plt.plot(px2, py2, label="Particle 2")
# plt.legend()
# plt.show()

avg = np.array([(px2+px1)/2, (py2+py1)/2, (pz2+pz1)/2])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(px1, py1, pz1, label="Particle 1")
ax.plot(px2, py2, pz2, label="Particle 2")
ax.plot(avg[0], avg[1], avg[2], label="Average")
ax.legend()

# Labels
ax.set_xlabel("X axis")
ax.set_ylabel("Y axis")
ax.set_zlabel("Z axis")

plt.show()