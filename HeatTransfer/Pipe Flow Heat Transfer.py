import numpy as np
import matplotlib.pyplot as plt

D_i = 0.02      # [m] Inner diameter
D_t = 0.01     # [m] Pipe thickness
D_o = D_i + 2*D_t # [m] Outer diameter

r_i = D_i / 2
r_o = D_o / 2
A_c = np.pi * r_i**2

r_a = r_o + 0.001

L = 0.5         # [m] Pipe length


# Coolant info
rho_c = 999.975    # [kg/m3] Fluid density
cp_c = 4187         # [J/kgK] Specific heat
mu_c = 6.5e-4       # [Pa-s] Dynamic viscosity
k_c = 0.63          # [W/m-K] Heat convection coefficient
Pr_c = cp_c * mu_c / k_c    # [] Prandtl Number

# Flow Conditions
u_c = 1.0           # [m/s] Flow velocity
mdot_c = rho_c * u_c * A_c
Re_c = rho_c * u_c * D_i / mu_c

# Coolant heat transfer coefficients
f = (0.79*np.log(Re_c) - 1.64)**-2
Nu_i = ((f/8)*(Re_c-1000) * Pr_c) / (1+12.7*np.sqrt(f/8)*(Pr_c**(2/3)-1))
h_c = Nu_i * k_c / D_i


# Wall conditions
h_w = 100.0         # [W/m2-K] Heat transfer coefficient
k_w = 16            # [W/m-K] Heat conduction coefficient
P_w = 2*np.pi*r_i   # [m] Inner wall perimeter

# Combined overall heat transfer coeff
def overall_U(h_c, h_w, k_w, r_i, r_o):
    u = 1 / ((1/h_c) + (np.log(r_o/r_i)/(k_w*2*np.pi*r_i))  + (r_i/(h_w*r_o)))
    return u

U = overall_U(h_c=h_c, h_w=h_w, k_w=k_w, r_i=r_i, r_o=r_o)

# Temperature init
Ti_c = 5 + 273.15  # [K] Coolant inlet temperature
Ti_w = 15 + 273.15  # [K] Init Wall temperature
T_a = 25 + 273.15   # [K] Outside air temp (const)



# === Grid setup ===
Nx = 2              # along pipe length
Ny = 200              # across full diameter (top to bottom)
x = np.linspace(0, L, Nx)
y = np.linspace(-r_a, r_a, Ny)
X, Y = np.meshgrid(x, y)

# === Temperature field initialization ===
T = np.ones_like(X) * T_a  # start with air temperature

# Assign regions
for j in range(Ny):
    for i in range(Nx):
        r = abs(Y[j, i])  # radial distance from centerline
        if r <= r_i:
            T[j, i] = Ti_c          # fluid region
        elif r_i < r <= r_o:
            T[j, i] = Ti_w          # pipe wall
        else:
            T[j, i] = T_a           # outer air region

# === Plot the full cross-section ===
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(
    T,
    origin='lower',
    cmap='inferno',
    extent=[0, L, -r, r],
    aspect='auto'
)

# Mark interfaces
ax.axhline(r_i, color='cyan', ls='--', lw=1.5, label='Fluid–Wall Boundary')
ax.axhline(-r_i, color='cyan', ls='--', lw=1.5)
ax.axhline(r_o, color='lime', ls='--', lw=1.5, label='Wall–Air Boundary')
ax.axhline(-r_o, color='lime', ls='--', lw=1.5)

# Labels & legend
plt.colorbar(im, label='Temperature [K]')
ax.set_xlabel('Pipe Length [m]')
ax.set_ylabel('Radial Distance [m]')
ax.set_title('Full Pipe Cross-Section: Air, Wall, and Fluid')
ax.legend(loc='upper right')
ax.grid(False)
plt.show()

