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

        if T_f > 510:
            T_f = 510

        if fluid == "RP-1":
            """https://rocketprops.readthedocs.io/en/latest/rp1_prop.html"""
            rho = 810.0 - 0.75 * (T_f - 288.15)                 # kg/m^3
            a0 = 9.06668
            a1 = -4.45988
            a2 = 164.814
            T_R = 1.8 * T_f
            mu = 10**a0 * T_R**a1 * 10**(a2/T_R)

            cp = (1.88192787e-05)*T_f**3 - (2.30240993e-02)*T_f**2 + (1.32103092e+01)*T_f - 3.12233222e+02
            k_btu = (-3.417e-11)*T_R**3 + (1.147e-7)*T_R**2 - (1.512e-4)*T_R + 1.183e-1
            k = 1.730735 * k_btu
            Pr = cp * mu / k

            gamma = 1.24 # (assumed)

            if not (1e-4 < mu < 5e-3):

                print(f"mu exceeds realistic value, clamping...")
                T_R = 500*1.8
                mu = 10**a0 * T_R**a1 * 10**(a2/T_R)


            dic[i]["T_max"] = 510



        elif coolant_only == False:

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

    info = {"CEA": True,
            "plots": "no",
            "dimensions": 1,    # Complexity of heat transfer
            "E": {
                "Pc": 3e6,  # Chamber Pressure [Pa]
                "Pe": 60000,  # Ambient Pressure (exit) [Pa]
                "Tc": 3500,  # Chamber temp [K]
                "mdot": 0.73,  # Mass Flow Rate [kg/s]
                "OF": 2.25,
                "size": 1.0,
                "CR": 3,
            },
            "H": {
                "mu": None,
                "k": None,
                "rho": None,
                "gamma": None,
                "cp": None,
                "cstar": None,
                "MW": None,
            },
            "F": {
                "Type": "RP-1",
                "T": 300,
                "P": None,
                "mu": None,
                "k": None,
                "rho": None,
                "gamma": None,
                "cp": None,
                "cstar": None,
                "MW": None,
                "mdot": None,
            },
            "O": {
                "Type": "LOX",
                "T": 98,
                "P": None,
                "mu": None,
                "k": None,
                "rho": None,
                "gamma": None,
                "cp": None,
                "cstar": None,
                "MW": None,
                "mdot": None,
            },
            "W": {
                # "Type": "SS 316L",
                # "Type": "Tungsten",
                "Type": "Copper Chromium",
                "thickness": 0.02
            },
            "C": {
                "Type": "Square",
                "spacing": 0.01,   # Fin thickness -- space between channels
                "height": 0.01,     # Channel height
                "num_ch": None,
                "h": None,
                "Nu": None,
                "Re": None,
            },
            "Flow": {
                "x": None,
                "y": None,
                "a": None,
                "eps": None
            },


            }
    #
    # Fluid_Properties(dic=info, coolant_only=True)
    # print(info["F"]["mu"])

    import numpy as np
    import matplotlib.pyplot as plt

    # Temperature sweep for RP-1 coolant
    T_vals = np.linspace(280, 510, 200)  # K (liquid RP-1 useful range)

    mu_arr = []
    rho_arr = []
    cp_arr = []
    k_arr = []
    Pr_arr = []

    for T in T_vals:
        info["F"]["T"] = T
        info["F"]["P"] = 3e6  # keep pressure fixed just for property trends

        Fluid_Properties(dic=info, coolant_only=True)

        mu_arr.append(info["F"]["mu"])
        rho_arr.append(info["F"]["rho"])
        cp_arr.append(info["F"]["cp"])
        k_arr.append(info["F"]["k"])
        Pr_arr.append(info["F"]["Pr"])

    mu_arr = np.array(mu_arr)
    rho_arr = np.array(rho_arr)
    cp_arr = np.array(cp_arr)
    k_arr = np.array(k_arr)
    Pr_arr = np.array(Pr_arr)

    fig, axs = plt.subplots(3, 2, figsize=(10, 10))
    axs = axs.flatten()

    axs[0].plot(T_vals, mu_arr)
    axs[0].set_title("RP-1 Viscosity μ [Pa·s]")
    axs[0].set_xlabel("T [K]")
    axs[0].set_ylabel("μ")

    axs[1].plot(T_vals, rho_arr)
    axs[1].set_title("RP-1 Density ρ [kg/m³]")
    axs[1].set_xlabel("T [K]")
    axs[1].set_ylabel("ρ")

    axs[2].plot(T_vals, cp_arr)
    axs[2].set_title("RP-1 cp [J/kg·K]")
    axs[2].set_xlabel("T [K]")
    axs[2].set_ylabel("cp")

    axs[3].plot(T_vals, k_arr)
    axs[3].set_title("RP-1 Thermal Conductivity k [W/m·K]")
    axs[3].set_xlabel("T [K]")
    axs[3].set_ylabel("k")

    axs[4].plot(T_vals, Pr_arr)
    axs[4].set_title("RP-1 Prandtl Number")
    axs[4].set_xlabel("T [K]")
    axs[4].set_ylabel("Pr")

    for ax in axs:
        ax.grid(True)

    plt.tight_layout()
    plt.show()
