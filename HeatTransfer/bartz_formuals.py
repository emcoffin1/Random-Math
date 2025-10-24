import numpy as np


def bartz_heat_transfer(gamma, R, Pc, rt, x, y, cp, k, mu, Tc, T, M, t_wall=0.002, k_wall=30, T_cool=500, h_c=15000,
                        Tw=900, Pr=0.9, w=0.7):
    """
    Uses Bartz Formula to compute the gas-side convective heat transfer coefficient
    Essentially shows you how efficiently the hot gas transfers heat to the wall
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
    c_star = np.sqrt(R * Tc) / gamma * (2/ (gamma + 1))**-((gamma + 1)/2/(gamma - 1))
    D = 2*y
    Dt = 2*rt

    # Initial wall temperature guess (through full engine)
    T_wall_inner = np.ones_like(x) * 800
    T_wall_innter_old = T_wall_inner.copy()
    T_wall_outer = np.zeros_like(x) * T_cool

    for i in range(1):
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
        q_dot = (Taw - T_cool) / R

        """Now solve for the wall temperatures, remembering that qdot = h(Taw-Tw)
        and qdot = h_c(Tw,o - T_air) --> this is the convection portion
        """
        # Inner wall
        T_wall_inner = Taw - (q_dot/hg)

        # Outer wall (for coolant)
        T_wall_outer = T_cool + (q_dot/h_c)

        dT = T_wall_inner - T_wall_outer
        print(dT)
        print(q_dot*t_wall/k_wall)

    return {"hg": hg, "qdot": q_dot, "T_wi": T_wall_inner, "T_wo": T_wall_outer}
