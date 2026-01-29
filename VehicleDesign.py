from rocketcea.cea_obj import CEA_Obj
import numpy as np
import matplotlib.pyplot as plt
import re
from CoolProp.CoolProp import PropsSI, AbstractState
import CoolProp.CoolProp as CP

def parse_formula(formula):
    tokens = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    elems = {}
    for el, n in tokens:
        elems[el] = elems.get(el, 0) + (int(n) if n else 1)
    return elems

def stoich_of_formula(fuel_formula, fuel_mw=None, oxid_formula="O2", oxid_mw=32.0):
    fuel = parse_formula(fuel_formula)

    c_mass = 12.011
    h_mass = 1.008
    o_mass = 16.0
    n_mass = 28.02

    C = fuel.get("C", 0)
    H = fuel.get("H", 0)
    O = fuel.get("O", 0)
    N = fuel.get("N", 0)

    # O atoms required for complete oxidation
    O_req = 2*C + 0.5*H + 2*N - O
    if O_req <= 0:
        raise ValueError("Fuel contains excess oxygen.")

    # kmol O2 per kmol fuel
    kmol_O2 = O_req / 2

    fuel_mw = (c_mass * C) + (h_mass * H) + (o_mass * O) + (n_mass * N)
    # Mass-based O/F
    OF = (kmol_O2 * oxid_mw) / fuel_mw
    return OF


def isp_getter(fuel, ox, plot=False):

    CEA = CEA_Obj(
        oxName=ox,
        fuelName=fuel)

    result_isp = []
    OF = np.linspace(1, 5, 500)
    pc = 2.5e6 * 0.000145038
    for i in OF:
        isp, _ = CEA.estimate_Ambient_Isp(Pc=pc, MR=i)
        result_isp.append(isp)
        # result_isp.append(CEA.get_Isp(Pc=pc, MR=i))


    idx = np.argmax(result_isp)
    of_best = OF[idx]
    print(f"OF for Maximum ISP: {of_best:.2f}")


    MAP = {
        "RP1": "C12H26",
        "Kerosene": "C12H26",
        "CH4": "CH4",
        "Methane": "CH4"
    }
    fuel = MAP.get(fuel, fuel)
    # of_stoich = stoich_of_formula("C12H26")
    of_stoich = stoich_of_formula(fuel)
    # print(f"Stoich OF: {of_stoich:.2f}")
    #
    if plot:
        plt.plot(OF, result_isp)
        plt.axvline(of_stoich, color='r', label="Stoich")
        plt.axvline(of_best, color='g', label="Best")
        plt.axvline(2.1, color="b", label="Ours")
        plt.legend()
        plt.show()

    return of_best, result_isp[idx]


def rocket_eqn_analysis(fuel, ox, alt: int, of, isp, T_W_ratio: float = 6, m0: float = 100):
    CEA = CEA_Obj(
        oxName=ox,
        fuelName=fuel)

    T_sl = T_W_ratio * m0 * 9.81
    mdot = T_sl / (isp * 9.81)

    mdot_fuel = mdot / (of + 1)
    mdot_lox = of * mdot / (1 + of)

    v_exit = T_sl / mdot

    max_exit_diameter = 6 * 0.0254
    max_exit_area = np.pi * max_exit_diameter ** 2 / 4

    P_exit = 101325

    # Iterate over Pressure ratios
    # Get mach number
    # With new chamber pressure compute gamma, R, T to get mach wrt v_exit
    # Compare difference, determine Pc that gives closest to v_exit
    Pc_list = np.linspace(P_exit, 9e6, 100)
    dif_list = []
    for i in Pc_list:
        Pr = i/P_exit

        Tc = CEA.get_Tcomb(Pc=i, MR=of) * 5/9
        mw, gam_c = CEA.get_Chamber_MolWt_gamma(Pc=i, MR=of)
        R = 8314.462618 / mw

        P = Pr
        gr = (gam_c - 1) / gam_c
        term1 = 2 / (gam_c - 1)
        term2 = P ** gr - 1
        M_i = np.sqrt(term1 * term2)
        # M_i = ((Pr**((gam_c - 1)/gam_c) - 1) * 2 / (gam_c - 1))**0.5
        T_static = Tc / (1 + ((gam_c - 1)/2*M_i**2))


        a = np.sqrt(gam_c * R * T_static)
        M_from_exit = v_exit / a

        dif_list.append(M_from_exit - M_i)
        # print(M_from_exit - M_i)

    # plt.plot(Pc_list, dif_list)
    # plt.show()

    Pc_chosen = 2.5e6
    mw, gam_c = CEA.get_Chamber_MolWt_gamma(Pc=Pc_chosen, MR=of)
    Tc = CEA.get_Tcomb(Pc=Pc_chosen, MR=of)
    R = 8314.462618 / mw
    P = Pc_chosen / P_exit
    gr = (gam_c - 1) / gam_c
    term1 = 2 / (gam_c - 1)
    term2 = P ** gr - 1
    M_at_pc = np.sqrt(term1 * term2)

    T_static = Tc / (1 + ((gam_c-1)/2*M_at_pc**2))
    aspect_ratio = (1/M_at_pc) * ((2/(gam_c+1)) * (1 + (((gam_c-1)/2)*M_at_pc**2)))**((gam_c+1)/(2*(gam_c-1)))

    A_throat = max_exit_area / aspect_ratio
    D_throat = np.sqrt(A_throat*4/np.pi)

    A_throat_choked = (mdot/Pc_chosen) * np.sqrt(T_static * R / gam_c) * (2/(gam_c+1))**((gam_c+1)/((2*(gam_c-1))))
    D_throat_choked = np.sqrt(A_throat_choked*4/np.pi)

    a = np.sqrt(gam_c * R * T_static)
    v_exit_actual = M_at_pc * a

    F = mdot * v_exit_actual



    print(f"SL Thrust due to T/W ratio: \n"
          f"{T_sl:.2f} N\n"
          f"{T_sl*0.224809:.2f} lbs\n"
          )
    print(f"At OF: {of:.2f}\n")
    print(f"Required mass flow rate: \n"
          f"{mdot:.2f} kg/s\n"
          f"{mdot * 2.20462:.2f} lb/s\n")
    print(f"Fuel mass flow rate: \n"
          f"{mdot_fuel:.2f} kg/s\n"
          f"{mdot_fuel * 2.20462:.2f} lb/s\n")
    print(f"Lox mass flow rate: \n"
          f"{mdot_lox:.2f} kg/s\n"
          f"{mdot_lox * 2.20462:.2f} lb/s\n")
    print(f"Target exhaust velocity: \n"
          f"{v_exit:.2f} m/s\n"
          f"{v_exit * 3.28084:.2f} ft/s\n")
    print(f"Chosen chamber pressure: \n"
          f"{Pc_chosen:.2f} Pa\n"
          f"{Pc_chosen * 0.000145038:.2f} PSI\n")
    print(f"Exit mach/velocity: \n"
          f"{M_at_pc:.2f} \n"
          f"{v_exit_actual:.2f} m/s\n"
          f"{v_exit_actual * 3.28084:.2f} ft/s\n")
    print(f"Aspect Ratio: \n"
          f"{aspect_ratio:.2f} \n")
    print(f"Throat area from ratio:\n"
          f"{A_throat:.2f} m^2\n"
          f"{A_throat*1550:.2f} in^2\n")
    print(f"Throat area from choked flow:\n"
          f"{A_throat_choked:.2f} m^2\n"
          f"{A_throat_choked*1550:.2f} in^2\n")
    print(f"Throat diameter from ratio:\n"
          f"{D_throat:.2f} m\n"
          f"{D_throat / 0.0254:.2f} in\n")
    print(f"Throat diameter from choked flow:\n"
          f"{D_throat_choked:.2f} m\n"
          f"{D_throat_choked / 0.0254:.2f} in\n")
    print(f"Actual force generated:\n"
          f"{F:.2f} N\n"
          f"{F*0.224809:.2f} lbf\n")



if __name__ == "__main__":
    # isp_getter(fuel="Ethanol", ox="O2", plot=True)
    # of, isp = isp_getter(fuel="Kerosene", ox="LOX")
    # rocket_eqn_analysis(fuel="Kerosene", ox="LOX", alt=1, isp=isp, of=of)

    cea = CEA_Obj(oxName="O2", fuelName="Kerosene")
    # print(cea.get_Chamber_Transport(Pc=305, MR=1.8)[2] * 1.7307)
    print(cea.get_Chamber_MolWt_gamma(Pc=305, MR=1.8)[0])

    # fluid = CP.AbstractState("HEOS", 'n-Dodecane')
    # fluid.update(CP.PT_INPUTS, 2.569e6, 300)
    # print(fluid.cpmass())
    # print(fluid.molar_mass())
    # print(8314.462618 / fluid.molar_mass()*1000)
    # print(fluid.conductivity())
    # print(fluid.viscosity())
    # for i in np.linspace(270,800, 20):
        #     print(cea.get_Throat_Transport(Pc=i, MR=1.8)[1] * 1e-4)
        # print(cea.get_Tcomb(Pc=217, MR=1.8)*5/9)

        # print(cea.get_HeatCapacities(Pc=273, MR=1.8)[1])

        # fluid.update(CP.PT_INPUTS, 3e6, i)
        # print(fluid.cpmass())
        # print(i)