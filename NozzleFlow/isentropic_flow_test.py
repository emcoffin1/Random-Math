from species import *
from extra_utils import mass_to_moles, moles_to_mass


R_u = 8.314462618  # J/(mol·K)


def chamber_state(P0, T0, gibbs, b_elem, species):
    """
    Computes the initial chamber properties
    :param P0: Chamber pressure
    :param T0: Chamber temperature
    :param gibbs: Gibbs minimizer object
    :param b_elem: Element list
    :param species: Species object
    :return:
    """
    # Determine the gas properties at the initial state
    # Get molar composition
    y = gibbs.solve(T=T0, P=P0, b_elem=b_elem)

    # Convert the molar to mass comp to use in gas prop equations
    Y = moles_to_mass(y, species)

    # Calculate the gas properties
    h, cp, R, gamma, W = mix_props_mass(T=T0, Y=Y, species=species)

    return {"h0": h, "cp": cp, "R": R, "gamma": gamma, "W": W, "Y": Y}


def mix_props_mass(T, Y, species):
    """
    Computes the properties of the combusted gas
    :param T: Temp [K]
    :param Y: Specific mass [kg]
    :param species: Species object
    :return: h, cp, R, gamma, W
    """
    # Initialize variables
    h = 0.0
    cp = 0.0
    inv_W = 0.0  # sum(Y_i / W_i)

    for name, Yi in Y.items():
        sp = species[name]
        Wi = sp.mw
        if Wi <= 0.0:
            raise ValueError(f"Species {name} has non-positive MW!")

        # Compute enthalpy and cp
        h += Yi * sp.h_mass(T)
        cp += Yi * sp.cp_mass(T)
        inv_W += Yi / Wi

    # Guard against 0 and compute the gas properties
    inv_W = max(1e-30, inv_W)
    W = 1 / inv_W
    R = R_u / W

    denom = max(cp - R, 1e-30)
    gamma = cp / denom

    return h, cp, R, gamma, W


def st_from_P_A_mdot_h(P, A, mdot, h_t, Y_frozen, b_elem, species, gibbs,
                       T_guess=3000, frozen=False, iter_max=300):
    # Initialize the temperature for the solver using the T guess
    T = float(T_guess)

    # Initialize the residual trackers
    f_prev, T_prev = 0, 0
    converged = False
    residual_final = np.nan
    aux_final = None

    def residual(T):
        # Get the mass fractions
        if not frozen:
            # If gas dynamics are varying
            y = gibbs.solve(T=T, P=P, b_elem=b_elem)
            Y = moles_to_mass(y=y, species=species)
        else:
            # If gas dynamics DONT vary
            Y = Y_frozen

        # Get all gas props
        h, cp, R, gamma, _ = mix_props_mass(T=T, Y=Y, species=species)

        # Compute the rho, v, mdot
        # Velocity from energy ke = h_t-h --> v = sqrt(2 KE)
        ke = max(0.0, h_t-h)
        v = (2 * ke) ** 0.5

        # rho from ideal gas
        rho = P / max(R * T, 1e-30)

        # mdot residual (d-mdot)
        f = mdot - rho * A * v

        # Return residual and new properties
        return f, (rho, v, h, cp, R, gamma, Y)


    # Loop
    for it in range(iter_max):
        # Get residual and gas products
        f, aux = residual(T)

        # Save the residual and aux just in case it's good, or we run out of iters
        residual_final = abs(f)
        aux_final = aux

        # Now check if we've converged
        if abs(f) < 1e-6 * max(1.0, abs(h_t)):
            # We've converged!
            converged = True

            # Separate the aux values to store
            rho, v, h, cp, R, gamma, Y = aux

            # And return a dictionary of useful info
            return {

            }


        # If not converged, adjust the values slightly in a secant-method fashion
        if f_prev is None:
            T_next = T * (0.95 if f > 0 else 1.05)

        else:
            dT, df = T - T_prev, f - f_prev

            # Check to make sure these won't break anything
            if not np.isfinite(df) or abs(df) < 1e-9 or not np.isfinite(f):
                T_next = T * (0.95 if f > 0 else 1.05)

            else:
                T_newton_step = T - f * (dT/df)
                # Damp the update to keep it stable
                T_next = 0.6 * T_newton_step + 0.4 * T

        # Clamp values for safety
        T_min, T_max = 300, 8000
        dT_max = 400
        T_next = np.clip(T_next, T - dT_max, T + dT_max)
        T_next = min(max(T_next, T_min), T_max)

        # And update the values for another iteration
        f_prev, T_prev = f, T
        T = T_next

    # If reaching this point, we failed to converge
    # Return last evaluated state along with residual stuff, set everything else to nan
    out = {

    }
    if aux_final is not None:
        # If we actually got some data out of it save and send it
        out.update({})
    return















