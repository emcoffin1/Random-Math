import numpy as np
from extra_utils import ConvergencePlot

def chamber_state(T0, P0, gibbs, b_elem, species):
    y = gibbs.solve(T=T0, P=P0, b_elem=b_elem)
    Y_mass = moles_to_mass(y, species=species)
    h0, cp, R, gamma, Wmix = mix_props_mass(T0, Y_mass, species)
    return {"h0": h0, "cp": cp, "R": R, "gamma": gamma, "Wmix": Wmix, "Y": Y_mass}


def state_from_PAmdotht(P, A, mdot, h_t, frozen, gibbs, b_elem, species, Y_frozen,
                        T_guess=3000.0, live_plot=True, max_iteration=60):
    T = T_guess
    f_prev, T_prev = None, None
    plotter = None

    if live_plot:
        title = f"T convergence at P={P:.2e} Pa"
        plotter = ConvergencePlot(title=title)

    def residual(T):
        """
        Computes the difference between the total enthalpy at (T,P) and the target total enthalpy h_t.
        """
        # Determine mass fractions from Gibbs minimization (unless frozen, then use previous)
        if not frozen:
            y_moles = gibbs.solve(T=T, P=P, b_elem=b_elem)
            y_vec = np.array([y_moles.get(sp.name) for sp in species])
            y = (y_vec / y_vec.sum())
            y = {sp.name: float(y[i]) for i, sp in enumerate(species)}
            print(f"Gibbs molar fractions @T={T:.0f}K P={P:.2e}:", y)
            Y = moles_to_mass(y, species=species)
        else:
            y = None
            Y = Y_frozen        # Inherits a mass based composition

        h, cp, R, gamma, _ = mix_props_mass(T, Y, species)
        rho = P / (R * T)
        v = mdot / (rho * A)
        f = (h + 0.5 * v * v) - h_t
        return f, (rho, v, h, cp, R, gamma, Y)

    converged = False
    residual_final = np.nan

    # ========== #
    # == LOOP == #
    # ========== #
    for it in range(max_iteration):
        f, aux = residual(T)
        residual_final = abs(f)
        if live_plot:
            plotter.update(it, f)

        if abs(f) < 1e-6 * abs(h_t):
            converged = True
            rho, v, h, cp, R, gamma, Y = aux
            if live_plot:
                plotter.close()
            return {"T": T, "P": P, "rho": rho, "v": v, "h": h,
                    "cp": cp, "R": R, "gamma": gamma, "Y": Y,
                    "converged": converged, "iterations": it+1,
                    "residual": residual_final}

        # Newton–Secant update
        if f_prev is None:
            T_next = T * (0.95 if f > 0 else 1.05)
        else:
            dT, df = T - T_prev, f - f_prev
            if not np.isfinite(df) or abs(df) < 1e-9 or not np.isfinite(f):
                T_next = T * (0.95 if f > 0 else 1.05)
            else:
                T_next = T - f * (dT / df)
            T_next = 0.6 * T_next + 0.4 * T

        T_prev, f_prev = T, f
        T = T_next

    if live_plot:
        plotter.close()

    return {"T": T, "P": P, "rho": np.nan, "v": np.nan, "h": np.nan,
            "cp": np.nan, "R": np.nan, "gamma": np.nan, "Y": None,
            "converged": False, "iterations": max_iteration, "residual": residual_final}




def solve_state(Ai, mdot, h_t, frozen, gibbs, b_elem, species, Y_frozen,
                P_lo, P_hi, T_hint, tol=1e-4, max_iter=60, max_expand=6):
    """
    Find flow state that matches given mdot and total enthalpy using adaptive pressure bracketing + bisection.
    """

    def mdot_error(P):
        """Return (error, state) for given pressure P."""
        try:
            st = state_from_PAmdotht(P, Ai, mdot, h_t, frozen, gibbs, b_elem,
                                     species, Y_frozen, T_guess=T_hint, live_plot=False)
            if not st["converged"]:
                print(f"Non-converged state at P={P:.2e} Pa (residual={st['residual']:.2e})")
            err = st["rho"] * st["v"] * Ai - mdot
            return err, st
        except RuntimeError:
            # Return a large error to force bracket expansion
            return np.inf, None

    # --- Evaluate initial bracket ---
    fa, sta = mdot_error(P_lo)
    fb, stb = mdot_error(P_hi)

    # --- Expand bracket if sign not opposite ---
    for expand in range(max_expand):
        if np.sign(fa) != np.sign(fb):
            break
        P_lo *= 0.8
        P_hi *= 1.2
        fa, sta = mdot_error(P_lo)
        fb, stb = mdot_error(P_hi)
    else:
        # No valid bracket — return whichever endpoint is closer
        if abs(fa) < abs(fb):
            return sta
        else:
            return stb

    # --- Bisection search ---
    for _ in range(max_iter):
        P_mid = 0.5 * (P_lo + P_hi)
        fm, stm = mdot_error(P_mid)

        if not np.isfinite(fm):
            fm = np.inf

        # Check convergence
        if abs(fm / mdot) < tol:
            return stm

        # Narrow the bracket
        if np.sign(fa) != np.sign(fm):
            P_hi, fb = P_mid, fm
        else:
            P_lo, fa = P_mid, fm

        # Optional small optimization: break early if bracket width tiny
        if abs(P_hi - P_lo) / P_mid < 1e-4:
            return stm

    # --- Return last computed state if no convergence ---
    return stm




def mix_props_mass(T, Y_mass, species: dict):
        """
        Compute mass based mixture properties from species list and mass fractions.
        :param T:
        :param Y_mass:
        :param species:
        :return:
        """
        # Initialize mixture properties
        h_mix = 0.0
        cp_mix = 0.0
        inv_Wmix = 0.0

        # Loop over species to compute mixture properties
        for name, Yi in Y_mass.items():
            sp = species[name]

            h_mix += Yi * sp.h_mass(T)
            cp_mix += Yi * sp.cp_mass(T)
            inv_Wmix += Yi / sp.mw

        Wmix = 1.0 / inv_Wmix
        R = 8.314462618 / Wmix  # J/mol-K
        gamma = cp_mix / (cp_mix - R)
        return h_mix, cp_mix, R, gamma, Wmix


def moles_to_mass(Y_mol, species):
    """
    Convert mole fractions to mass fractions.
    """
    w = {}
    denom = sum(Y_mol[k]*species[k].mw for k in Y_mol)
    for k in Y_mol:
        w[k] = Y_mol[k]*species[k].mw/denom
    return w


def mass_to_moles(Y_mass, species):
    """
    Convert mass fractions to mole fractions.
    """
    x = {}
    denom = sum(Y_mass[k]/species[k].mw for k in Y_mass)
    for k in Y_mass:
        x[k] = Y_mass[k]/species[k].mw/denom
    return x
