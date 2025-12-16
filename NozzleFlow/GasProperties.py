from rocketcea.cea_obj import CEA_Obj
from scipy.interpolate import PchipInterpolator
from CoolProp.CoolProp import PropsSI
import numpy as np

def HotGas_Properties(Pc, fuel, ox, OF, dic, eps=None):
    """Passes in metric but is converted to SI. RocketCEA returns SI and is then converted to metric"""
    cea = CEA_Obj(
        oxName=ox,
        fuelName=fuel
    )
    Pc_psi = Pc * 0.000145038

    dic["E"]["Tc"] = cea.get_Tcomb(Pc=Pc_psi, MR=OF) * 5/9

    if eps is not None:
        mw, gam_c = cea.get_Chamber_MolWt_gamma(Pc=Pc_psi, MR=OF, eps=eps)
        cp = [v * 4186.8 for v in cea.get_HeatCapacities(Pc=Pc_psi, MR=OF, eps=eps)]
        gam_t = cea.get_Throat_MolWt_gamma(Pc=Pc_psi, MR=OF, eps=eps)[1]
        gam_e = cea.get_exit_MolWt_gamma(Pc=Pc_psi, MR=OF, eps=eps)[1]
    else:
        mw, gam_c = cea.get_Chamber_MolWt_gamma(Pc=Pc_psi, MR=OF)
        cp = [v * 4186.8 for v in cea.get_HeatCapacities(Pc=Pc_psi, MR=OF)]
        gam_t = cea.get_Throat_MolWt_gamma(Pc=Pc_psi, MR=OF)[1]
        gam_e = cea.get_exit_MolWt_gamma(Pc=Pc_psi, MR=OF)[1]


    cstar = cea.get_Cstar(Pc=Pc_psi, MR=OF) * 0.3048
    R = 8314.462618 / mw

    mu = [
        cea.get_Chamber_Transport(Pc=Pc_psi, MR=OF)[1] / 0.671968975,
        cea.get_Throat_Transport(Pc=Pc_psi, MR=OF)[1] / 0.671968975,
        cea.get_Exit_Transport(Pc=Pc_psi, MR=OF)[1] / 0.671968975,
    ]

    k = [
        cea.get_Chamber_Transport(Pc=Pc_psi, MR=OF)[2] / 0.000481055,
        cea.get_Throat_Transport(Pc=Pc_psi, MR=OF)[2] / 0.000481055,
        cea.get_Exit_Transport(Pc=Pc_psi, MR=OF)[2] / 0.000481055,
    ]
    Pr = [
        cea.get_Chamber_Transport(Pc=Pc_psi, MR=OF)[3],
        cea.get_Throat_Transport(Pc=Pc_psi, MR=OF)[3],
        cea.get_Exit_Transport(Pc=Pc_psi, MR=OF)[3]
    ]

    m = [0.01, 1.0, cea.get_MachNumber(Pc=Pc_psi, MR=OF)]
    M_tab = np.array(m)
    mu = np.array(mu)
    k = np.array(k)
    cp = np.array(cp)
    gamma = np.array([gam_c, gam_t, gam_e])

    # Interpolators
    mu_M = PchipInterpolator(M_tab, mu)
    k_M = PchipInterpolator(M_tab, k)
    cp_M = PchipInterpolator(M_tab, cp)
    gamma_M = PchipInterpolator(M_tab, gamma)

    # Expand to flowfield if present
    if "M" in dic["Flow"]:

        M = dic["Flow"]["M"]
        dic["H"]["mu"] = mu_M(M)
        dic["H"]["cp"] = cp_M(M)
        dic["H"]["gamma"] = gamma_M(M)
        dic["H"]["k"] = k_M(M)


    else:

        dic["H"]["mu"] = mu
        dic["H"]["cp"] = cp
        dic["H"]["gamma"] = gam_t
        dic["H"]["k"] = k

    dic["H"]["Pr"] = Pr
    dic["H"]["cstar"] = cstar
    dic["H"]["R"] = R
    dic["H"]["MW"] = mw




def Fluid_Properties(dic: dict):
    """Currently assumes ideal conditions and ideal gas/fluids"""
    mdot = dic["E"]["mdot"]
    of = dic["E"]["OF"]
    dic["F"]["mdot"] = mdot / (of + 1)
    dic["O"]["mdot"] = of * mdot / (of + 1)

    FLUID_MAP = {
        "LOX": "Oxygen",
        "GOX": "Oxygen",
        "RP-1": "RP-1",
        "Kerosene": "RP-1",
        "Kero": "RP-1",
    }


    for i in ["F", "O"]:
        fluid = dic[i]["Type"]
        fluid = FLUID_MAP.get(fluid, fluid)

        P_f = dic[i]["P"] if dic[i]["P"] is not None else dic["E"]["Pc"] + 689476
        T_f = dic[i]["T"] if dic[i]["T"] is not None else 298

        if fluid == "RP-1":
            A = -7.812
            B = 5.53e3
            C = -1.503e6
            D = 1.801e8
            invT = 1 / T_f
            ln_nu = A + B*invT + C*(invT**2) + D*(invT**3)
            nu = 10**ln_nu * 1e-6                               # m^2/s
            rho = 810.0 - 0.75 * (T_f - 288.15)                 # kg/m^3
            mu = rho * nu                                       # Pa-s

            cp = 2000
            k = 0.13
            Pr = cp * mu / k

            gamma = 1.24 # (assumed)

            if not (1e-4 < mu < 5e-3):
                raise ValueError(
                    f"Unphysical RP-1 Viscosity: mu = {mu:.3f} [Pa-s] at T = {T_f:.2f} [K]"
                )


        else:

            rho = PropsSI("D", "T", T_f, "P", P_f, fluid)
            mu = PropsSI("V", "T", T_f, "P", P_f, fluid)
            cp = PropsSI("C", "T", T_f, "P", P_f, fluid)
            k = PropsSI("L", "T", T_f, "P", P_f, fluid)
            cv = PropsSI("O", "T", T_f, "P", P_f, fluid)
            gamma = cp / cv
            Pr = cp * mu / k

        dic[i]["mu"] = mu
        dic[i]["rho"] = rho
        dic[i]["cp"] = cp
        dic[i]["k"] = k
        dic[i]["Pr"] = Pr
        dic[i]["gamma"] = gamma

def Material_Properties(dic: dict):
    """Manual input of multiple materials for easy reference"""
    material = dic["W"]["Type"]

    if material == "SS 304":
        cp = 500            # J/kg-K
        solidus = 1673      # K
        liquidus = 1728     # K
        rho = 8000          # kg/m^3
        yield_strength = 215e6  # Pa,
        E = 195e9           # Pa
        k = 16.2            # W/m-K

    elif material == "SS 316L":
        cp = 500            # J/kg-K
        solidus = 1648      # K
        liquidus = 1673     # K
        rho = 8000          # kg/m^3
        yield_strength = 205e6  # Pa
        E = 193e9           # Pa
        k = 16.2            # W/m-K

    elif material == "Inconel 718":
        cp = 435            # J/kg-K
        solidus = 1533      # K
        liquidus = 1609     # K
        rho = 8190          # kg/m^3
        yield_strength = 1100e6  # Pa
        E = 193e9           # Pa
        k = 11.4            # W/m-K

    elif material == "Tungsten":
        cp = 134  # J/kg-K
        solidus = 3695  # K
        liquidus = 3697  # K
        rho = 19300  # kg/m^3
        yield_strength = 750e6  # Pa
        E = 300e9  # Pa
        k = 117  # W/m-K

    elif material == "Copper Chromium":
        cp = 390  # J/kg-K
        solidus = 1290  # K
        liquidus = 1350  # K
        rho = 8900  # kg/m^3
        yield_strength = 120e6  # Pa
        E = 110e9  # Pa
        k = 300  # W/m-K


    else:
        cp = None
        solidus = None
        liquidus = None
        rho = None
        yield_strength = None
        k = None
        E = None

    dic["W"]["cp"] = cp
    dic["W"]["solidus"] = solidus
    dic["W"]["liquidus"] = liquidus
    dic["W"]["rho"] = rho
    dic["W"]["yield_strength"] = yield_strength
    dic["W"]["k"] = k
    dic["W"]["E"] = E

