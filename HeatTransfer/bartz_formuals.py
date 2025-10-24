import numpy as np


def bartz_heat_transfer(gamma, R, Pc, rt, x, y, cp, k, mu, Tc, T, M, t_wall=0.002, k_wall=30, T_cool=298,
                        h_c=15000, w=0.7):
    """
    Uses Bartz Formula to compute the gas-side convective heat transfer coefficient
    Essentially shows you how efficiently the hot gas transfers heat to the wall.
    qdot is the heat flux per unit area, or how much heat is transferred per area of wall
    :param gamma:
    :param R:
    :param Pc:
    :param rt:
    :param x:
    :param y:
    :param cp:
    :param k:
    :param mu:
    :param Tc:
    :param T:
    :param Tw:
    :param Pr:
    :param w:
    :return:
    """
    # Variable Calcs and Assumptions
    c_star = (1/np.sqrt(gamma)) * ((2/(gamma + 1))**((gamma + 1)/(2*(gamma - 1)))) * np.sqrt(R * Tc)

    D = 2*y
    Dt = 2*rt

    # Initial wall temperature guess (through full engine)
    Tw_init = 0.45 * Tc
    T_wall_inner = np.ones_like(x) * Tw_init

    for i in range(50):
        T_film = 0.5 * (T + T_wall_inner)

        hg = (0.026 / (Dt**0.2)) \
            * (Pc / c_star)**0.8 \
            * (Dt / D)**1.8 \
            * cp * (mu**0.2) \
            * (T / T_film)**(0.8 - 0.2 * w)
        """Knowing hg, we can now solve for the heat flux
        We do this by looking at the thermal resistance through all stages
        In this section, we only care about gas side to wall conduction to ambient
        Later we will use the to determine the coolant required
        Rg = 1/hg
        Rw = tw/kw
        Ra = 1/ha
        R = Rg + Rw + Rc
        """

        R_total = (1/hg) + (t_wall/k_wall) + (1/h_c)

        """The thermal heat flux, q''=(T_aw - T_bulk)/R, 
        shows the heat flux by the resistance. T_aw is the adiabatic wall temperature,
        is found using the recovery-temperature formula. This requires r=Pr^1/3, and Pr= u c_p/k
         
        """
        r = (mu*cp/k)**(1/3)
        Taw = T*(1 + r*(gamma-1)/2*M**2)
        q_dot_res = (Taw - T_cool) / R_total
        q_dot_con = hg*(Taw - T_wall_inner)

        """Now solve for the wall temperatures, remembering that qdot = h(Taw-Tw)
        and qdot = h_c(Tw,o - T_air) --> this is the convection portion
        """
        # Inner wall
        T_wall_inner_new = Taw - (q_dot_res/hg)
        # Temperature relaxation to check difference
        # The relaxation automatically adjusts the next iteration
        T_wall_inner = 0.5*T_wall_inner_new + 0.5*T_wall_inner

        # Outer wall (for coolant)
        T_wall_outer = T_cool + (q_dot_res/h_c)

        dic = {"hg": hg, "qdot": q_dot_res, "T_wi": T_wall_inner, "T_wo": T_wall_outer,
               "t_wall": t_wall, "T_ci": T_cool}

        dT = np.max(np.abs(T_wall_inner_new - T_wall_inner))
        if dT < 1e-3:
            print(f"Heat transfer converged in {i} iterations!")
            return dic

    return dic


def total_heat_flux(qdot, x, y, cp, Tc_in, Tc_out):
    """
    Computes the total heat flux to the wall of the engine at each step
    :param qdot:
    :param x:
    :param y:
    :return: {Q, Qtotal, mdot}
    """
    """Compute the first pass mdot"""
    # Determine the perimeter (s) of each step
    # Uses 2*pi*r
    p = 2*np.pi*y

    # Determine the effective surface area for each step
    dx = np.gradient(x)
    sa = p * dx

    # Find Q of each section
    Q = qdot * sa

    # Find Qtotal
    Qtot = np.sum(Q)

    # Find mdot
    mdot = Qtot / cp / (Tc_out - Tc_in)

    return {"Q": Q, "Qtotal": Qtot, "mdot": mdot}