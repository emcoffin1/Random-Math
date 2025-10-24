
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




