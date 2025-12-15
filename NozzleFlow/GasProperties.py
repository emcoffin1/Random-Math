from rocketcea.cea_obj import CEA_Obj
from math import exp
from CoolProp.CoolProp import PropsSI

def HotGas_Properties(Pc, fuel, ox, OF, dic, eps=None):
    """Passes in metric but is converted to SI. RocketCEA returns SI and is then converted to metric"""
    cea = CEA_Obj(
        oxName=ox,
        fuelName=fuel
    )
    Pc = Pc * 0.000145038
    if eps is not None:
        dic["H"]["MW"], dic["H"]["gamma"] = cea.get_Chamber_MolWt_gamma(Pc=Pc, MR=OF, eps=eps)
        dic["H"]["cp"] = cea.get_Chamber_Cp(Pc=Pc, MR=OF, eps=eps) * 4186.8
    else:
        dic["H"]["MW"], dic["H"]["gamma"] = cea.get_Chamber_MolWt_gamma(Pc=Pc, MR=OF)
        dic["H"]["cp"] = cea.get_Chamber_Cp(Pc=Pc, MR=OF) * 4186.8


    dic["E"]["Tc"] = cea.get_Tcomb(Pc=Pc, MR=OF) * 5/9
    dic["H"]["cstar"] = cea.get_Cstar(Pc=Pc, MR=OF) * 0.3048
    dic["H"]["R"] = 8314.462618 / dic["H"]["MW"]

    dic["H"]["k"] = [v / 0.000481055 for v in [cea.get_Chamber_Transport(Pc=Pc, MR=OF)[2], cea.get_Throat_Transport(Pc=Pc, MR=OF)[2], cea.get_Exit_Transport(Pc=Pc, MR=OF)[2]]]
    dic["H"]["mu"] = [v / 0.671968975 for v in [cea.get_Chamber_Transport(Pc=Pc, MR=OF)[1], cea.get_Throat_Transport(Pc=Pc, MR=OF)[1], cea.get_Exit_Transport(Pc=Pc, MR=OF)[1]]]
    dic["H"]["Pr"] = [cea.get_Chamber_Transport(Pc=Pc, MR=OF)[3], cea.get_Throat_Transport(Pc=Pc, MR=OF)[3], cea.get_Exit_Transport(Pc=Pc, MR=OF)[3]]

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
            print(rho)
            if not (1e-4 < mu < 5e-3):
                raise ValueError(
                    f"Unphysical RP-1 Viscosity: mu = {mu:.3f} [Pa-s] at T = {T_f:.2f} [K]"
                )

            dic[i]["mu"] = mu
            dic[i]["rho"] = rho
            dic[i]["cp"] = 2000
            dic[i]["k"] = 0.13

        else:

            dic[i]["rho"] = PropsSI("D", "T", T_f, "P", P_f, fluid)
            dic[i]["mu"] = PropsSI("V", "T", T_f, "P", P_f, fluid)
            dic[i]["cp"] = PropsSI("C", "T", T_f, "P", P_f, fluid)
            dic[i]["k"] = PropsSI("L", "T", T_f, "P", P_f, fluid)
            cv = PropsSI("O", "T", T_f, "P", P_f, fluid)
            dic[i]["gamma"] = dic[i]["cp"] / cv
