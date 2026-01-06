from rocketcea.cea_obj import CEA_Obj
from scipy.interpolate import PchipInterpolator
from CoolProp.CoolProp import PropsSI
import CoolProp.CoolProp as CP
import numpy as np
from rocketprops.rocket_prop import get_prop

def init_cea(data):
    data["CEA_obj"] = CEA_Obj(
        oxName=data["O"]["Type"],
        fuelName=data["F"]["Type"],
    )
    data["rp1_prop_obj"] = get_prop("RP1")

def HotGas_Properties(dic, eps=None, forced=False, channel=False):
    """Passes in metric but is converted to SI. RocketCEA returns SI and is then converted to metric"""
    Pc, fuel, ox, OF = dic["E"]["Pc"], dic["F"]["Type"], dic["O"]["Type"], dic["E"]["OF"]
    cea = dic["CEA_obj"]
    Pc_psi = Pc * 0.000145038

    Tc = dic["E"]["Tc"]
    if Tc is None:
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
        cea.get_Chamber_Transport(Pc=Pc_psi, MR=OF)[1] * 1e-4,
        cea.get_Throat_Transport(Pc=Pc_psi, MR=OF)[1] * 1e-4,
        cea.get_Exit_Transport(Pc=Pc_psi, MR=OF)[1] * 1e-4,
    ]

    k = [
        cea.get_Chamber_Transport(Pc=Pc_psi, MR=OF)[2] * 0.4184,
        cea.get_Throat_Transport(Pc=Pc_psi, MR=OF)[2] * 0.4184,
        cea.get_Exit_Transport(Pc=Pc_psi, MR=OF)[2] * 0.4184,
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


    # Expand to flowfield if present
    if "M" in dic["Flow"] or forced:

        # Interpolators
        mu_M = PchipInterpolator(M_tab, mu)
        k_M = PchipInterpolator(M_tab, k)
        cp_M = PchipInterpolator(M_tab, cp)
        gamma_M = PchipInterpolator(M_tab, gamma)

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

    if channel:

        dic["H"]["mu"] = mu
        dic["H"]["cp"] = cp
        dic["H"]["gamma"] = gam_t
        dic["H"]["k"] = k

    dic["H"]["Pr"] = Pr
    dic["H"]["cstar"] = cstar
    dic["H"]["R"] = R
    dic["H"]["MW"] = mw




def Fluid_Properties(dic: dict, coolant_only=False):
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
    if coolant_only:
        fluids = ["F"]
    else:
        fluids = ["F", "O"]

    for i in fluids:
        fluid = dic[i]["Type"]
        fluid = FLUID_MAP.get(fluid, fluid)

        P_f = dic[i]["P"] if dic[i]["P"] is not None else dic["E"]["Pc"] + 689476
        T_f = dic[i]["T"] if dic[i]["T"] is not None else 298
        # print("NOW HERE",T_f)

        if fluid == "RP-1":
            # R = 49  # J/kg/k
            # """https://rocketprops.readthedocs.io/en/latest/rp1_prop.html"""
            # rho = 810.0 - 0.75 * (T_f - 288.15)                 # kg/m^3
            # a0 = 9.06668
            # a1 = -4.45988
            # a2 = 164.814
            T_R = 1.8 * T_f
            # print("NOWNOWHERE HERE",T_R)
            # mu = 10**a0 * T_R**a1 * 10**(a2/T_R)
            #
            # cp = (1.88192787e-05)*T_f**3 - (2.30240993e-02)*T_f**2 + (1.32103092e+01)*T_f - 3.12233222e+02
            # k_btu = (-3.417e-11)*T_R**3 + (1.147e-7)*T_R**2 - (1.512e-4)*T_R + 1.183e-1
            # k = 1.730735 * k_btu
            # Pr = cp * mu / k
            #
            # gamma = 1.24 # (assumed)
            #
            # if not (1e-4 < mu < 5e-3):
            #
            #     print(f"mu exceeds realistic value, clamping...")
            #     T_R = 500*1.8
            #     mu = 10**a0 * T_R**a1 * 10**(a2/T_R)
            #
            #
            # dic[i]["T_max"] = 510

            # pObj = dic["rp1_prop_obj"]
            pObj = get_prop("RP-1")
            cp = pObj.CpAtTdegR(T_R) * 4186
            cpv = PropsSI("Cpmass", "T", T_f, "P", P_f, "n-Dodecane")
            k = pObj.CondAtTdegR(T_R) * 1.730735
            mu = pObj.ViscAtTdegR(T_R) * 0.1
            rho = 999.016 * pObj.SGLiqAtTdegR(T_R)
            Pr = cp * mu / k
            R = None
            gamma = None



        elif coolant_only == False:

            R_u = PropsSI("GAS_CONSTANT", "T", T_f, "P", P_f, fluid)
            M = PropsSI("molemass", "T", T_f, "P", P_f, fluid)
            R = R_u / M
            rho = PropsSI("D", "T", T_f, "P", P_f, fluid)
            mu = PropsSI("V", "T", T_f, "P", P_f, fluid)
            cp = PropsSI("C", "T", T_f, "P", P_f, fluid)
            cpv = PropsSI("Cpmass", "T", T_f, "P", P_f, fluid)
            k = PropsSI("L", "T", T_f, "P", P_f, fluid)
            cv = PropsSI("O", "T", T_f, "P", P_f, fluid)
            gamma = cp / cv
            Pr = cp * mu / k

        dic[i]["R"] = R
        dic[i]["mu"] = mu
        dic[i]["rho"] = rho
        dic[i]["cp"] = cp
        dic[i]["cp_v"] = cpv
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
        k = 20            # W/m-K

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


if __name__ == "__main__":

    fluids = CP.get_global_param_string("FluidsList")
    print(fluids)
