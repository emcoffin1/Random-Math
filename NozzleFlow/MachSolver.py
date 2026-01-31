
import numpy as np
from scipy.optimize import brentq

def area_ratio_from_M(M, gamma=1.4):
    g = gamma
    return (1.0 / M) * ((2 / (g + 1) * (1 + (g - 1) / 2 * M**2)) ** ((g + 1) / (2 * (g - 1))))


def mach_from_area_ratio(eps, gamma=1.4):
    """
    Returns (M_sub, M_sup) for a given area ratio A/A* = eps.
    Handles the sonic case and bracketing robustly.
    """
    eps_c = eps.copy()
    eps = abs(eps)
    if eps <= 0:
        raise ValueError("Area ratio must be positive.")

    # If A/A* = 1, flow is sonic
    if np.isclose(eps, 1.0, atol=1e-6):
        return 1.0

    # Define residual
    def F(M): return area_ratio_from_M(M, gamma) - eps

    # --- Subsonic branch ---
    M_sub = np.nan
    if eps > 1.0:  # Only valid if A/A* > 1
        try:
            M_sub = brentq(F, 1e-6, 0.9999, maxiter=100)
        except ValueError:
            pass  # if no valid root, leave as nan

    # --- Supersonic branch ---
    M_sup = np.nan
    try:
        M_sup = brentq(F, 1.0001, 50.0, maxiter=100)
    except ValueError:
        pass

    if eps_c < 1.0:
        return M_sub
    else:
        return M_sup


def isentropic_nozzle_flow(eps, data: dict):
    # Unpack data dictionary
    T0, P0, gamma, R, H0 = data["E"]['Tc'], data["E"]['Pc'], data["H"]['gamma'], data["H"]['R'], data["H"]["H"]
    At, Ae, Pa = np.min(data["E"]["a"]), data["E"]["a"][-1], data["E"]["Pe"]

    # Compute mach through geometry
    M = np.array([mach_from_area_ratio(eps=e, gamma=gamma) for e in eps])
    # Compute static temp
    T = T0 / (1 + (gamma-1)/2 * M**2)
    # Compute static pressure
    P = P0 * (T/T0)**(gamma/(gamma-1))
    # Compute speed of sound at each station
    a = np.sqrt(gamma * R * T)
    # Compute speed of gas
    U = M * a
    # Compute density
    rho = P / (R * T)
    # Compute Enthalpy Values
    H = H0 - 0.5*U**2

    # Double check mdot
    cf_mom = np.sqrt(
        (2 * gamma ** 2 / (gamma - 1)) *
        (2 / (gamma + 1)) ** ((gamma + 1) / (gamma - 1)) *
        (1 - (P[-1] / P0) ** ((gamma - 1) / gamma))
    )
    # pressure contribution
    cf_pres = (P[-1] - Pa) / P0 * (Ae / At)
    c_f = cf_pres + cf_mom
    # print(c_f)
    cstar = data["H"]["cstar"]
    mdot = P0 * At / cstar
    # print(mdot)

    data["E"]["mdot"] = mdot
    data["E"]["c_f"] = c_f
    data["Flow"]["M"] = M
    data["Flow"]["P"] = P
    data["Flow"]["T"] = T
    data["Flow"]["U"] = U
    data["Flow"]["rho"] = rho
    data["Flow"]["H"] = H



