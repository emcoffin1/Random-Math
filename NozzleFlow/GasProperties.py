from rocketcea.cea_obj import CEA_Obj
from scipy.interpolate import PchipInterpolator
from CoolProp.CoolProp import PropsSI, AbstractState
import CoolProp.CoolProp as CP
import numpy as np
from rocketprops.rocket_prop import get_prop

"""
https://coolprop.org/coolprop/HighLevelAPI.html#reference-states
"""

def init_cea(data):
    data["CEA_obj"] = CEA_Obj(
        oxName=data["O"]["Type"],
        fuelName=data["F"]["Type"],
    )
    cea = CEA_Obj(oxName="O2", fuelName="Kerosene")
    cea.get_Chamber_H()

    FLUID_MAP = {
        "LOX": "Oxygen",
        "GOX": "Oxygen",
        "RP-1": "n-Dodecane",
        "Kerosene": "n-Dodecane",
        "Kero": "n-Dodecane",
        "CH4": "Methane"
    }

    for key in ["F", "O"]:
        fluid = FLUID_MAP.get(data[key]["Type"], data[key]["Type"])
        data[key]["State"] = AbstractState("HEOS", fluid)


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

    H = cea.get_Chamber_H(Pc=Pc_psi, MR=OF) * 2.326


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
    dic["H"]["MW"] = mw*0.453592
    dic["H"]["H"] = H




def Fluid_Properties(dic: dict, coolant_only=False):
    """Currently assumes ideal conditions and ideal gas/fluids"""
    energy_method = dic["Solver"]["EnergyMethod"]

    mdot = dic["E"]["mdot"]
    of = dic["E"]["OF"]
    dic["F"]["mdot"] = mdot / (of + 1)
    dic["O"]["mdot"] = of * mdot / (of + 1)

    FLUID_MAP = {
        "LOX": "Oxygen",
        "GOX": "Oxygen",
        "RP-1": "n-Dodecane",
        "Kerosene": "n-Dodecane",
        "Kero": "n-Dodecane",
        "CH4": "Methane"
    }

    fluids = ["F"] if coolant_only else ["F", "O"]

    for i in fluids:
        fluid = dic[i]["Type"]
        try:
            fluid = FLUID_MAP.get(fluid, fluid)
        except:
            pass
        st = dic[i]["State"]

        P_f = dic[i]["P"] if dic[i]["P"] is not None else dic["E"]["Pc"] + 689476

        if energy_method:
            H = dic[i]["H"]

            s = CP.PropsSI("S", "P", P_f, "H", H, fluid)
            st.update(CP.PSmass_INPUTS, P_f, s)

        else:
            T = dic[i]["T"]
            st.update(CP.PT_INPUTS, P_f, T)


        # Saturation guard
        Tsat = hsat = None
        if P_f < st.p_critical():
            try:
              st.update(CP.PQ_INPUTS, P_f, 0.0)
              Tsat = st.T()
              hsat = st.hmass()
            except ValueError:
                pass

        # ---- Restore actual state after PQ call ----
        if energy_method:
            st.update(CP.PSmass_INPUTS, P_f, s)
        else:
            st.update(CP.PT_INPUTS, P_f, T)

        dic[i]["H"] = st.hmass()
        dic[i]["T"] = st.T()
        dic[i]["Tcrit"] = st.T_critical()

        dic[i]["Tsat"] = Tsat
        dic[i]["Hsat"] = hsat

        # props = [m for m in dir(st) if not m.startswith("_")]
        #
        # for p in sorted(props):
        #     print(p)

        dic[i]["R"] = st.gas_constant() / st.molar_mass()
        dic[i]["mu"] = st.viscosity()
        dic[i]["rho"] = st.rhomass()
        dic[i]["cp"] = st.cpmass()
        dic[i]["cpv"] = st.cpmass()
        dic[i]["cv"] = st.cvmass()
        dic[i]["k"] = st.conductivity()
        dic[i]["Pr"] = st.Prandtl()
        dic[i]["gamma"] = st.cpmass() / st.cvmass()

        if not np.isfinite(dic["F"]["k"]):
            raise ValueError(f"BAD k at: \n"
                  f"{P_f} Pa\n"
                  f"{dic["F"]["H"]} J/kg")


def Material_Properties(dic: dict, T: float = None):
    """Manual input of multiple materials for easy reference"""
    material = dic["W"]["Type"]
    if T is None:
        T = dic["W"]["T"] # K
    else:
        T = T

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

    elif material == "GRCop-42":
        """https://additivemanufacturingllc.com/wp-content/uploads/2023/03/GRCop-42.pdf"""
        """https://velo3d.com/wp-content/uploads/2024/04/Velo3D-GRCop-42-Material-Datasheet.pdf"""
        """https://www.sciencedirect.com/science/article/pii/S2352492823013569"""
        if T is not None:
            # These are accurate to around 1000K,
            # everything after is extrapolated and assumed to fit on the same curve
            T = T / 1000
            cp = 0.153*T**3 - 0.33*T**2 + 0.312*T + 0.316
            k = -39.86 * T ** 3 + 4.17 * T ** 2 + 0.97 * T + 329.14
        else:
            cp = 0  # J/kg-K
            k = 315  # W/m-K
        solidus = 0  # K
        liquidus = 0  # K
        rho = 8890  # kg/m3
        yield_strength = 3.585e+8  # Pa
        E = 129.7e9  # Pa
        roughness = 2e-5  # m


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
