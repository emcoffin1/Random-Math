import numpy as np
from NozzleFlow.GasProperties import Fluid_Properties

def bartz_heat_transfer_const(info: dict, w=0.7):
    """
    Uses Bartz Formula to compute the gas-side convective heat transfer coefficient
    Essentially shows you how efficiently the hot gas transfers heat to the wall.
    qdot is the heat flux per unit area, or how much heat is transferred per area of wall

    This function currently assumes a constant coolant temperature
    :param info: Dictionary of all things
    """
    # Dictionary breakdown
    Pc, Rt, gamma, R, k, Tc, mdot, mu, cp, k, cstar = (info["E"]["Pc"], info["E"]["Rt"], info["H"]["gamma"], info["H"]["R"],
                                                info["H"]["k"],
                                                info["E"]["Tc"], info["E"]["mdot"], info["H"]["mu"][1],
                                                info["H"]["cp"][1], info["H"]["k"][1], info["H"]["cstar"])
    dittus_appro(dic=info, dimension=0)
    mdot_f, cp_f, T_f, h_f = info["F"]["mdot"], info["F"]["cp"], info["F"]["T"], info["F"]["h"]

    t_w, k_w = info["W"]["thickness"], info["W"]["k"]

    x, y, T, M = info["Flow"]["x"], info["Flow"]["y"], info["Flow"]["T"], info["Flow"]["M"]

    D = 2*y
    Dt = 2*Rt

    for i in range(1):

        # Adiabatic wall temperature calculation
        Pr = info["H"]["Pr"][1]
        r = Pr ** (1 / 3)
        T0 = Tc  # chamber stagnation; assume constant
        Taw = T0 * (1 + r * (gamma - 1) / 2 * M ** 2) / (1 + (gamma - 1) / 2 * M ** 2)

        h_h = (0.026 / (Dt**0.2)) \
            * (Pc / cstar)**0.8 \
            * (Dt / D)**1.8 \
            * cp * (mu**0.2) \
            * (T / Taw)**(0.8 - 0.2 * w)

        """Knowing hg, we can now solve for the heat flux
        We do this by looking at the thermal resistance through all stages
        In this section, we only care about gas side to wall conduction to ambient
       Through the wall, and then 
        Rc = 1/hc
        Rw = tw/kw
        R = Rg + Rw + Rc
        """

        # Rg = 1 / hg
        Rw = t_w / k_w
        Rf = 1 / h_f
        Rh = 1 / h_h
        R_total = Rf + Rw + Rh

        """
        Find the initial wall temperature
        Use initial wall temperature to determine the heat flux (q")
        """
        q_conv  = (Taw - T_f) / R_total
        # T_wall_initial = (T_f + h_h * (Rw + Rf) * Taw) / (1 + h_h * (Rw + Rf))

        # q" (heat flux) W/m^2
        # q_conv = h_h * (Taw - T_wall_initial)
        T_wall_initial = Taw - q_conv * Rh

        dic = {"hg": h_h, "qdot": q_conv, "T_wi": T_wall_initial, "T_aw": Taw, "T_cool": np.ones_like(T)*T_f}

    return dic


def bartz_heat_transfer_1d(info: dict, w=0.7):

    """Engine and Hot Gas Information"""
    # Dictionary breakdown
    Pc, Rt, gamma, R, Tc, mdot, cstar = (info["E"]["Pc"], info["E"]["Rt"], info["H"]["gamma"],
                                                       info["H"]["R"], info["E"]["Tc"],
                                                       info["E"]["mdot"], info["H"]["cstar"])
    mu, k, cp = info["H"]["mu"], info["H"]["k"], info["H"]["cp"]

    mdot_f, cp_f, T_f = info["F"]["mdot"], info["F"]["cp"], info["F"]["T"]

    t_w, k_w = info["W"]["thickness"], info["W"]["k"]

    x: list = info["Flow"]["x"]
    y: list = info["Flow"]["y"]
    T: list = info["Flow"]["T"]
    M: list = info["Flow"]["M"]

    D = 2 * y
    Dt = 2 * Rt

    N = len(x)  # Number of x points

    """Coolant and Wall Information"""
    P_w = np.pi * D     # Wetted perimeter

    """Storage"""
    Taw = np.zeros(N)
    h_h = np.zeros(N)
    T_w = np.zeros(N)
    q_conv = np.zeros(N)
    T_c = np.zeros(N)

    """Initial Conditions"""
    T_c[-1] = T_f

    """March along x"""
    for i in range(N-1, -1, -1):


        dittus_appro(dic=info, dimension=1)
        h_f = info["F"]["h"]

        # Hot Gas properties
        Pr = mu[i] * cp[i] / k[i]
        r_rec = Pr ** (1/3)

        # Adiabatic wall temperature
        Taw[i] = Tc * (1 + r_rec * (gamma[i]-1)/2 * M[i]**2) / (1 + (gamma[i]-1)/2 * M[i]**2)

        # Bartz convection coefficient approximation
        h_h[i] = (0.026 / (Dt**0.2)) \
                * (Pc / cstar)**0.8 \
                * (Dt / D[i])**1.8 \
                * cp[i] * (mu[i]**0.2) \
                * (Tc / Taw[i])**(0.8 - 0.2 * w)


        Rw = t_w / k_w
        Rf = 1 / h_f
        Rh = 1 / h_h[i]
        R_total = Rf + Rw + Rh

        # Wall temperature from resistance network
        q_conv[i] = (Taw[i] - T_c[i]) / R_total
        T_wall_initial = Taw[i] - q_conv[i] * Rh

        T_w[i] = T_wall_initial

        # Update coolant temp at next station
        if i > 0:
            dx = x[i] - x[i-1]
            dTc = q_conv[i] * P_w[i] * dx / (mdot_f * cp_f)
            T_c[i-1] = T_c[i] + dTc

    dic = {"hg": h_h, "qdot": q_conv, "T_wi": T_w, "T_aw": Taw, "T_cool": T_c}
    return dic


def dittus_appro(dic:dict, dimension: int):
    """Computes convective heat transfer coefficient for the coolant in channels"""
    # Update some fluid properties if necessary
    # Only for 0 dimensional/constant coolant temp
    if dimension != 0:
        Fluid_Properties(dic=dic)

    rho, mu, k, cp, Pr, mdot = (dic["F"]["rho"], dic["F"]["mu"], dic["F"]["k"], dic["F"]["cp"],
                                dic["F"]["Pr"], dic["F"]["mdot"])
    Dh, ch_num = dic["E"]["Dh"], dic["E"]["Channel_num"]

    Ah = np.pi * (Dh/2)**2

    # Using mass flux to avoid using velocity
    if mdot is not None and Ah is not None:
        G = mdot / (Ah * ch_num)
        Re = G * Dh / mu
    else:
        raise ValueError("Need fuel mass flow rate and channel area (flow area) "
                         "to determine convective heat transfer coefficient")

    if Re < 2300:
        # Fully-developed laminar, constant wall heat flux
        Nu = 4.36

    else:
        # Gnielinksi (for smooth duct)
        f = (0.79 * np.log(Re) - 1.64) ** -2
        Nu = ((f/8.0) * (Re - 1000) * Pr) / (1.0 + 12.7 * np.sqrt(f/8) * (Pr**(2/3) - 1.0))

    h_f = Nu * k / Dh
    dic["F"]["h"] = h_f
    dic["F"]["Nu"] = Nu
    dic["F"]["Re"] = Re

