from MachSolver import mach_from_area_ratio_supersonic as mach_eps
import numpy as np
import matplotlib.pyplot as plt
from NozzleDesign import build_nozzle
import extra_utils as utils


def nozzle_flow(eps, T0, P0, gamma, R):

    M = np.array([mach_eps(eps=e, gamma=gamma) for e in eps])
    T = T0 / (1 + (gamma-1)/2 * M**2)
    P = P0 * (T/T0)**(gamma/(gamma-1))
    a = np.sqrt(gamma * R * T)
    U = M * a
    rho = P / (R * T)
    return M, U, T, P, rho


def main(Rt, T0, P0, gamma, R, cp, k, mu):
    # Get nozzle geometry
    x, y, a = build_nozzle(Pe=101300, Pc=P0, size=0.8, Rt=Rt, gamma=gamma, plots="no")
    At = np.pi * Rt**2
    eps = a / At

    # Get flow properties
    M, U, T, P, rho = nozzle_flow(eps, T0, P0, gamma, R)

    # Plot flow nozzle values
    l = [M, U, T, P, rho]
    labels = ["Mach Number", "Velocity [m/s]", "Temperature [K]", "Pressure [Pa]", "Density [kg/m³]"]
    # utils.plot_flow_char(x=x, data=l, labels=labels)
    # utils.plot_flow_field(x, y, T, "Temp", mode=2)
    utils.convert_to_func(x,y)


if __name__ == '__main__':
    Ri = 0.05  # m (throat radius)

    T0 = 2774.0   # K
    P0 = 2.013e6  # Pa
    gamma = 1.2013
    Rgas = 421.6  # J/kg-K

    # Properties (SI)
    cp = 2852.7  # J/kg-K  (2.8527 kJ/kg-K)
    k = 0.5937  # W/m-K
    mu = 8.617e-4  # Pa·s  (if you meant 0.8617 mPa·s)

    main(Rt=Ri, T0=T0, P0=P0, gamma=gamma, R=Rgas, k=k, mu=mu, cp=cp)
