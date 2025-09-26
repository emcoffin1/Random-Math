import numpy as np
import matplotlib.pyplot as plt

Ft = 4448.22        # Thrust (N)
Cd = 0.371          # Drag coefficient
Ac = 0.381          # Cross-sectional area (m^2)
rho = 1.161         # Air density (kg/m^3)
m_prop = 22.67      # mass of fuel (kg)
t_burn = 15              # burn time (s)
mf = 72.53          # initial rocket mass (kg)
g = 9.81            # gravity (m/s^2)
mdot = m_prop/t_burn     # mass flow rate (kg/s)
Ve = Ft / mdot      # effective exhaust velocity (m/s)
me = mf - m_prop    # Empty mass (kg)

# Init variable
# time, alt, velocity, gravity
t, h, v, m = 0, 0, 0, mf
dt = 0.001

# Storage
time, alt, vel, mom = [],[],[],[]

while True:
    # Drag at current velocity
    D = 0.5 * v**2 * rho * Ac * Cd

    # Thrust
    T = Ft if t < t_burn else 0.0

    # Mass change
    if t < t_burn:
        m = m - mdot*dt
    else:
        m = me


    # acceleration
    sum_forces = -D + T
    a = -g + sum_forces/m

    # Euler step
    v += a*dt
    h += v*dt
    t += dt

    # Save
    time.append(t)
    alt.append(h)
    vel.append(v)
    mom.append(m*v)

    # Break if apogee
    if t > t_burn and h <= 0:
        break


print(f"Max altitude = {max(alt):.2f}m")
print(f"Max velocity = {max(vel):.2f}m/s")

# plt.plot(time, mom, label='Momentum Change')
plt.plot(time, alt, label='Alt')
plt.show()

