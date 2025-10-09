import numpy as np
from extra_utils import ConvergencePlot
from species import get_species_data

def chamber_state(T0, P0, gibbs, b_elem, species):
    y = gibbs.solve(T=T0, P=P0, b_elem=b_elem)
    Y_mass = moles_to_mass(y, species=species)
    h0, cp, R, gamma, Wmix = mix_props_mass(T0, Y_mass, species)
    return {"h0": h0, "cp": cp, "R": R, "gamma": gamma, "Wmix": Wmix, "Y": Y_mass}


def _normalize_fractions(d):
    out = {k: max(float(v), 0.0) for k, v in d.items()}
    s = sum(out.values())
    if s <= 0.0:
        raise ValueError("Fractions sum to zero.")
    return {k: v/s for k, v in out.items()}


def _to_mole_fractions(n_dict):
    total = sum(n_dict.values())
    if total <= 0.0:
        raise ValueError("Total moles are zero.")
    return {k: v/total for k, v in n_dict.items()}


def state_from_PAmdotht(P, A, mdot, h_t, frozen, gibbs, b_elem, species, Y_frozen,
                        T_guess=3000.0, live_plot=False, max_iteration=60):
    T = float(T_guess)
    f_prev, T_prev = None, None
    plotter = ConvergencePlot(title=f"T convergence at P={P:.2e} Pa") if live_plot else None

    # return these if we exit without strict convergence
    last_aux = None
    residual_final = np.nan

    def residual(T):
        """
        Mass-flow residual at (T,P): f = mdot - rho(T,P)*v(T)*A,
        where v(T) comes from energy: v = sqrt(2*(h_t - h(T))).
        """
        # composition
        if not frozen:
            n_eq = gibbs.solve(T=T, P=P, b_elem=b_elem)  # moles
            y_mole = {k: v / sum(n_eq.values()) for k, v in n_eq.items()}
            Y = moles_to_mass(y_mole, species=species)  # mass fractions for props
        else:
            Y = _normalize_fractions(Y_frozen)

        # mixture properties
        h, cp, R, gamma, _ = mix_props_mass(T, Y, species)

        # velocity from energy (not from mdot!)
        ke = max(0.0, h_t - h)  # specific kinetic energy [J/kg]
        v = (2.0 * ke) ** 0.5

        # density from ideal gas
        rho = P / max(R * T, 1e-12)

        # mass-flow residual
        f = mdot - rho * v * A
        return f, (rho, v, h, cp, R, gamma, Y)

    # === Iteration ===
    converged = False
    for it in range(max_iteration):
        f, aux = residual(T)
        last_aux = aux
        residual_final = abs(f)
        if live_plot:
            plotter.update(it, f)

        if abs(f) < 1e-6 * max(1.0, abs(h_t)):
            converged = True
            if live_plot: plotter.close()
            rho, v, h, cp, R, gamma, Y = aux
            return {"T": T, "P": P, "rho": rho, "v": v, "h": h,
                    "cp": cp, "R": R, "gamma": gamma, "Y": Y,
                    "converged": True, "iterations": it+1, "residual": residual_final}

        # Secant-like update with safeguards
        if f_prev is None:
            # First step: small relative nudge based on sign
            T_next = T * (0.95 if f > 0 else 1.05)
        else:
            dT, df = T - T_prev, f - f_prev
            if not np.isfinite(df) or abs(df) < 1e-9 or not np.isfinite(f):
                # fallback step
                T_next = T * (0.95 if f > 0 else 1.05)
            else:
                T_newton = T - f * (dT / df)   # secant
                # damp the update (keeps it stable)
                T_next = 0.6 * T_newton + 0.4 * T

        # Clamp to physical, cap step
        T_min, T_max = 200.0, 6000.0
        max_step = 400.0
        T_next = np.clip(T_next, T - max_step, T + max_step)
        T_next = min(max(T_next, T_min), T_max)

        T_prev, f_prev = T, f
        T = T_next

    if live_plot:
        plotter.close()

    # Return the last evaluated state (best effort)
    out = {"T": T, "P": P, "rho": np.nan, "v": np.nan, "h": np.nan,
           "cp": np.nan, "R": np.nan, "gamma": np.nan, "Y": None,
           "converged": False, "iterations": max_iteration, "residual": residual_final}
    if last_aux is not None:
        rho, v, h, cp, R, gamma, Y = last_aux
        out.update({"rho": rho, "v": v, "h": h, "cp": cp, "R": R, "gamma": gamma, "Y": Y})
    return out





def solve_state(Ai, mdot, h_t, frozen, gibbs, b_elem, species, Y_frozen,
                P_lo, P_hi, T_hint, tol=1e-4, max_iter=60, max_expand=8):
    """
    Find (T,P,...) that matches mass flow and stagnation enthalpy.
    Uses adaptive bracketing + Illinois (modified regula falsi).
    Returns the best-known state even if not strictly converged.
    """

    if frozen:
        if Y_frozen is None:
            raise ValueError("Frozen mode requires Y_frozen.")
        # Normalize once; downstream routines re-normalize each call too
        Y_frozen = _normalize_fractions(Y_frozen)

    # keep last good state to return on failure
    best = None
    best_rel_err = np.inf

    def mdot_error(P, T_guess):
        """Return (error, state, T_out) for given P."""
        st = state_from_PAmdotht(P, Ai, mdot, h_t, frozen, gibbs, b_elem,
                                 species, Y_frozen, T_guess=T_guess, live_plot=False)
        # compute error even if not converged (if finite)
        if np.isfinite(st.get("rho", np.nan)) and np.isfinite(st.get("v", np.nan)):
            err = st["rho"] * st["v"] * Ai - mdot
        else:
            err = np.inf

        rel = abs(err) / max(1e-12, abs(mdot))
        nonlocal best, best_rel_err
        if rel < best_rel_err:
            best, best_rel_err = st, rel

        # if err is nan/inf, push it away in a direction that promotes bracketing
        if not np.isfinite(err):
            # crude sign heuristic: if P is low, assume mdot too low (err negative), else too high
            err = -np.inf if P < 0.5*(P_lo+P_hi) else np.inf

        # pass out a good next T guess (use the solved T if converged, else previous guess)
        T_out = st["T"] if st["converged"] and np.isfinite(st["T"]) else T_guess
        return err, st, T_out

    # ---------- Initial bracket ----------
    fa, sta, Ta = mdot_error(P_lo, T_hint)
    fb, stb, Tb = mdot_error(P_hi, T_hint)

    # Expand bracket until sign change or attempts exhausted
    for k in range(max_expand):
        if np.sign(fa) != np.sign(fb) and np.isfinite(fa) and np.isfinite(fb):
            break
        # Expand conservatively (avoid going negative)
        P_lo = max(1e-3, 0.7 * P_lo)
        P_hi = 1.3 * P_hi
        fa, sta, Ta = mdot_error(P_lo, Ta)
        fb, stb, Tb = mdot_error(P_hi, Tb)
    else:
        # No sign change; return endpoint with smaller relative error
        return sta if abs(fa)/max(1e-12, abs(mdot)) <= abs(fb)/max(1e-12, abs(mdot)) else stb

    # ---------- Illinois method inside bracket ----------
    Pa, Pb = P_lo, P_hi
    Ta_guess, Tb_guess = Ta, Tb

    for _ in range(max_iter):
        # Regulafalsi candidate
        Pm = Pb - fb * (Pb - Pa) / (fb - fa)
        # Keep inside [Pa, Pb]
        if not (min(Pa, Pb) < Pm < max(Pa, Pb)) or not np.isfinite(Pm):
            Pm = 0.5 * (Pa + Pb)  # fallback to bisection

        fm, stm, Tm_guess = mdot_error(Pm, 0.5*(Ta_guess + Tb_guess))

        # Convergence on relative mdot error
        if abs(fm) / max(1e-12, abs(mdot)) < tol:
            return stm

        # Update bracket with Illinois weighting
        if np.sign(fa) == np.sign(fm):
            # fa and fm same sign: replace 'a' and halve fb (Illinois trick)
            Pa, fa, Ta_guess = Pm, fm, Tm_guess
            fb *= 0.5
        else:
            # fb and fm same sign: replace 'b' and halve fa
            Pb, fb, Tb_guess = Pm, fm, Tm_guess
            fa *= 0.5

        # Early exit if bracket width is tiny
        if abs(Pb - Pa) / max(1.0, Pm) < 1e-5:
            return stm

    # No strict convergence: return best-known state
    return best if best is not None else stm



R_u = 8.314462618  # J/(mol·K)

def _normalize_fractions(d):
    """Return a copy of d normalized to sum=1, clipping tiny negatives."""
    out = {k: max(float(v), 0.0) for k, v in d.items()}
    s = sum(out.values())
    if s <= 0.0:
        raise ValueError("Fraction dictionary sums to zero or negative.")
    return {k: v/s for k, v in out.items()}

def mix_props_mass(T: float, Y_mass: dict, species: dict):
    """
    Mixture properties from MASS fractions.
    Returns:
        h_mix [J/kg], cp_mix [J/(kg·K)], R [J/(kg·K)], gamma [-], Wmix [kg/mol]
    """
    # Normalize & clip
    Y = _normalize_fractions(Y_mass)

    h_mix = 0.0
    cp_mix = 0.0
    inv_Wmix = 0.0  # 1/W_mix = sum(Y_i / W_i)

    for name, Yi in Y.items():
        sp = species[name]
        Wi = sp.mw  # kg/mol (ensure NASA MW / 1000 done at load)
        if Wi <= 0.0:
            raise ValueError(f"Species {name} has nonpositive MW.")
        h_mix += Yi * sp.h_mass(T)     # J/kg
        cp_mix += Yi * sp.cp_mass(T)   # J/(kg·K)
        inv_Wmix += Yi / Wi

    # Guard numerical issues
    inv_Wmix = max(inv_Wmix, 1e-30)
    Wmix = 1.0 / inv_Wmix            # kg/mol
    R = R_u / Wmix                   # J/(kg·K)  <-- note units!
    # Prevent division by tiny (cp - R) due to round-off
    denom = max(cp_mix - R, 1e-12)
    gamma = cp_mix / denom

    return h_mix, cp_mix, R, gamma, Wmix


def moles_to_mass(Y_mol: dict, species: dict):
    """
    Convert mole FRACTIONS or raw moles -> mass FRACTIONS.
    Accepts either; we normalize inside.
    """
    x = _normalize_fractions(Y_mol)

    # denominator: sum(x_i * W_i) = average MW (kg/mol)
    denom = 0.0
    for k, xi in x.items():
        denom += xi * species[k].mw
    denom = max(denom, 1e-30)

    w = {k: (x[k] * species[k].mw) / denom for k in x}
    # ensure exact normalization (avoid drift)
    return _normalize_fractions(w)


def mass_to_moles(Y_mass: dict, species: dict):
    """
    Convert mass FRACTIONS -> mole FRACTIONS.
    """
    Y = _normalize_fractions(Y_mass)

    # numerator_i = Y_i / W_i ; denom = sum(Y_i / W_i)
    numer = {k: (Y[k] / species[k].mw) for k in Y}
    denom = sum(numer.values())
    denom = max(denom, 1e-30)

    x = {k: numer[k] / denom for k in numer}
    return x
