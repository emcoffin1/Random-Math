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

    for i in ["F", "O"]:
        fluid = dic[i]["Type"]
        P_f = dic[i]["P"] if dic[i]["P"] is not None else dic["E"]["Pc"] - 689476
        T_f = dic[i]["T"] if dic[i]["T"] is not None else 298

        if fluid == "RP-1":
            A = 1.8e-6
            B = 900
            dic[i]["rho"] = 810.0 - 0.75 * (T_f - 288.15)
            dic[i]["mu"] = A * exp(-B * dic[i]["T"])
            dic[i]["cp"] = 2000
            dic[i]["k"] = 0.13

        else:
            dic[i]["rho"] = PropsSI("D", "T", T_f, "P", P_f, fluid)
            dic[i]["mu"] = PropsSI("V", "T", T_f, "P", P_f, fluid)
            dic[i]["cp"] = PropsSI("C", "T", T_f, "P", P_f, fluid)
            dic[i]["k"] = PropsSI("L", "T", T_f, "P", P_f, fluid)
