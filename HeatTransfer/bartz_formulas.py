import numpy as np

def bartz_heat_transfer(x, y, cp, T, M, info: dict,t_wall=0.002, k_wall=30, T_cool=298,
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
    Pc, Rt, gamma, R, k, mu, Tc, mdot = (info["Pc"], info["Rt"], info["gamma"], info["R"], info["k"], info["mu"],
                                         info["Tc"], info["mdot"])

    # Variable Calcs and Assumptions
    c_star = c_star_ideal(Tc=Tc, gamma=gamma, R=R)

    D = 2*y
    Dt = 2*Rt

    for i in range(1):

        # Adiabatic wall temperature calculation
        Pr = mu * cp / k
        r = Pr ** (1 / 3)
        T0 = Tc  # chamber stagnation; assume constant
        Taw = T0 * (1 + r * (gamma - 1) / 2 * M ** 2) / (1 + (gamma - 1) / 2 * M ** 2)

        hg = (0.026 / (Dt**0.2)) \
            * (Pc / c_star)**0.8 \
            * (Dt / D)**1.8 \
            * cp * (mu**0.2) \
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

        """The thermal heat flux, q''=(T_aw - T_bulk)/R, 
        shows the heat flux by the resistance. T_aw is the adiabatic wall temperature,
        is found using the recovery-temperature formula. This requires r=Pr^1/3, and Pr= u c_p/k
         
        """
        T_wall_initial = (T_cool + hg * (Rw + Rc) * Taw) / (1 + hg * (Rw + Rc))

        T_wall_arr = np.ones_like(T) * T_wall_initial
        # q" (heat flux) W/m^2
        q_conv = hg * (Taw - T_wall_arr)

        dic = {"hg": hg, "qdot": q_conv, "T_wi": T_wall_arr, "T_aw": Taw}

    return dic


def bartz_step(T_isen_stat, M, Tc, y, Dt, Pc, cp, k, mu, gamma=1.22, R=350, w=0.7):

    # C star using stagnation conditions, needed for bartz
    c_star = (1 / np.sqrt(gamma)) * ((2 / (gamma + 1)) ** ((gamma + 1) / (2 * (gamma - 1)))) * np.sqrt(R * Tc)

    # Engine diameter at station
    D = y*2

    # Find T film used in bartz
    r = (mu*cp/k)**(1/3)
    T_aw = T_isen_stat * (1 + r * (gamma-1) / 2 * M**2)
    T_film = 0.5 * (T_isen_stat + T_aw)

    hg = (0.026 / (Dt ** 0.2)) \
         * (Pc / c_star) ** 0.8 \
         * (Dt / D) ** 1.8 \
         * cp * (mu ** 0.2) \
         * (T_isen_stat / T_film) ** (0.8 - 0.2 * w)

    return hg



def total_heat_flux(qdot, x, y, cp, Tc_in, Tc_out,):
    """
    Computes the total heat flux to the wall of the engine at each step
    :param qdot:
    :param x:
    :param y:
    :return: {Q, Qtotal, mdot}
    """

    """
    Section out each channel region:
    converging -> throat -> diverging
    """
    eps = y / np.min(y)
    max_rat = 1.05
    div, thr, con = [], [], []

    for idx in range(1, len(eps)):
        i = eps[idx]
        prev = eps[idx-1]
        dA = i - prev
        if i > max_rat and dA > 0:
            div.append(i)
        elif i > max_rat and dA < 0:
            con.append(i)
        else:
            thr.append(i)

    np.array(div), np.array(thr), np.array(con)
    sections = [con, thr, div]


    """Find the Q of each station and the Q total"""
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

    return {"Q": Q, "Qtotal": Qtot}

def coolant_temp_solver(y, T_isen, M_isen, Pc, cp, k, mu, mdot, t_wall=0.002 ,k_wall=30, T_coolant=298, type="square"):

    # Throat diameter
    Dt = 2 * np.min(y)

    # Init coolant temp arrays
    T_cool = np.zeros_like(y)
    T_cool[-1] = T_coolant

    # Chamber Temp
    Tc = T_isen[0]

    # We will assume that the heat flux is constant and solve for each station
    # This needs to be changed later
    for i in range(len(T_cool) - 1):
        # Offset the i so we can go backwards
        i_in = -i - 1
        i_out = i_in - 1

        # Station info
        T_isen_stat = T_isen[i]
        y_stat = y[i]
        M_stat = M_isen[i]

        # Bartz heat transfer coeff and resistance
        hg_bartz = bartz_step(T_isen_stat=T_isen_stat, M=M_stat, y=y_stat, Dt=Dt, Pc=Pc, cp=cp, k=k, mu=mu, Tc=Tc)
        R_bartz = 1 / hg_bartz

        # Wall resistance
        R_wall = t_wall/k_wall

        # Coolant heat transfer and resistance
        T_in = T_cool[i_in]


def c_star_ideal(Tc, gamma, R):
    Gamma = np.sqrt(gamma) * (2.0/(gamma+1.0))**((gamma+1.0)/(2.0*(gamma-1.0)))
    return np.sqrt(R*Tc) / Gamma