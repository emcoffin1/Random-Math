import numpy as np

def bartz_heat_transfer_const(x, y, cp, T, M, info: dict,t_wall=0.002, k_wall=30, T_cool=298,
                        h_c=15000, w=0.7):
    """
    Uses Bartz Formula to compute the gas-side convective heat transfer coefficient
    Essentially shows you how efficiently the hot gas transfers heat to the wall.
    qdot is the heat flux per unit area, or how much heat is transferred per area of wall

    This function currently assumes a constant coolant temperature
    :param x:
    :param y:
    :param cp:
    :param T:
    :param w:
    :return:
    """
    # Dictionary breakdown
    Pc, Rt, gamma, R, mu, Tc, mdot = (info["E"]["Pc"], info["E"]["Rt"], info["H"]["gamma"], info["H"]["R"], info["H"]["mu"],
                                         info["E"]["Tc"], info["E"]["mdot"])

    # Variable Calcs and Assumptions
    c_star = info["H"]["cstar"] if not None else c_star_ideal(Tc=Tc, gamma=gamma, R=R)

    D = 2*y
    Dt = 2*Rt

    for i in range(1):

        # Adiabatic wall temperature calculation
        Pr = info["H"]["Pr"][1]
        r = Pr ** (1 / 3)
        T0 = Tc  # chamber stagnation; assume constant
        Taw = T0 * (1 + r * (gamma - 1) / 2 * M ** 2) / (1 + (gamma - 1) / 2 * M ** 2)

        hg = (0.026 / (Dt**0.2)) \
            * (Pc / c_star)**0.8 \
            * (Dt / D)**1.8 \
            * cp * (mu[1]**0.2) \
            * (T / Taw)**(0.8 - 0.2 * w)
        """Knowing hg, we can now solve for the heat flux
        We do this by looking at the thermal resistance through all stages
        In this section, we only care about gas side to wall conduction to ambient
        Later we will use the to determine the coolant required
        Rg = 1/hg
        Rw = tw/kw
        Ra = 1/ha
        R = Rg + Rw + Rc
        """

        # Rg = 1 / hg
        Rw = t_wall / k_wall
        Rc = 1 / h_c
        # R_total = Rg + Rw + Rc

        """
        Find the initial wall temperature
        Use initial wall temperature to determine the heat flux (q")
        """
        T_wall_initial = (T_cool + hg * (Rw + Rc) * Taw) / (1 + hg * (Rw + Rc))

        # q" (heat flux) W/m^2
        q_conv = hg * (Taw - T_wall_initial)

        dic = {"hg": hg, "qdot": q_conv, "T_wi": T_wall_initial, "T_aw": Taw}

    return dic


def bartz_heat_transfer_1d(x, y, cp, T, M, info: dict, t_wall=0.002, k_wall=30, T_cool=298,
                           h_c=15000, w=0.7):

    """Engine and Hot Gas Information"""
    # Dictionary breakdown
    Pc, Rt, gamma, R, k, Tc, mdot, mu, cp, k = (info["E"]["Pc"], info["E"]["Rt"], info["H"]["gamma"], info["H"]["R"], info["H"]["k"],
                                         info["E"]["Tc"], info["E"]["mdot"], info["H"]["mu"][1], info["H"]["cp"][1], info["H"]["k"][1])

    mdot_f = info["F"]["mdot"]
    # Variable Calcs and Assumptions
    c_star = info["H"]["cstar"] if not None else c_star_ideal(Tc=Tc, gamma=gamma, R=R)

    D = 2 * y
    Dt = 2 * Rt

    N = len(x)  # Number of x points

    """Coolant and Wall Information"""
    P_w = np.pi * D     # Wetted perimeter

    """Storage"""
    Taw = np.zeros(N)
    hg = np.zeros(N)
    T_w = np.zeros(N)
    q_conv = np.zeros(N)
    T_c = np.zeros(N)

    """Initial Conditions"""
    T_c[0] = T_cool

    Rw = t_wall / k_wall
    Rc = 1 / h_c

    """March along x"""
    for i in range(N):
        # Gas properties
        Pr = mu * cp / k
        r_rec = Pr ** (1/3)

        # Adiabatic wall temperature
        Taw[i] = Tc * (1 + r_rec * (gamma-1)/2 * M[i]**2) / (1 + (gamma-1)/2 * M[i]**2)

        # Bartz convection coefficient approximation
        hg[i] = (0.026 / (Dt**0.2)) \
                * (Pc / c_star)**0.8 \
                * (Dt / D[i])**1.8 \
                * cp * (mu**0.2) \
                * (T[i] / Taw[i])**(0.8 - 0.2 * w)

        # Wall temperature from resistance network
        Tw_i = (T_c[i] + hg[i] * (Rw + Rc) * Taw[i]) / (1.0 + hg[i] * (Rw + Rc))
        T_w[i] = Tw_i

        # Heat flux at this station
        q_conv[i] = hg[i] * (Taw[i] - T_w[i])

        # Update coolant temp at next station
        if i < N-1:
            dx = x[i+1] - x[i]
            dTc = q_conv[i] * P_w[i] * dx / (mdot_c * cp_c)
            T_c[i+1] = T_c[i] + dTc

    dic = {"hg": hg, "qdot": q_conv, "T_wi": T_w, "T_aw": Taw, "T_cool": T_c}
    return dic



def c_star_ideal(Tc, gamma, R):
    Gamma = np.sqrt(gamma) * (2.0/(gamma+1.0))**((gamma+1.0)/(2.0*(gamma-1.0)))
    return np.sqrt(R*Tc) / Gamma