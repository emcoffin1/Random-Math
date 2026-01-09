from rocketcea.cea_obj import CEA_Obj
import numpy as np
import matplotlib.pyplot as plt
import re

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

    result = []
    OF = np.linspace(1, 5, 100)
    for i in OF:
        result.append(CEA.get_Isp(Pc=300, MR=i))

    idx = np.argmax(result)
    print(f"OF for Maximum ISP: {OF[idx]:.2f}")


    MAP = {
        "RP1": "C12H26",
        "Kerosene": "C12H26",
        "CH4": "CH4",
        "Methane": "CH4"
    }
    fuel = MAP.get(fuel, fuel)
    # of_stoich = stoich_of_formula("C12H26")
    of_stoich = stoich_of_formula(fuel)
    print(f"Stoich OF: {of_stoich:.2f}")
    #
    if plot:
        plt.plot(OF, result)
        plt.axvline(of_stoich, color='r')
        plt.show()
    return OF[idx], result[idx]


def rocket_eqn_analysis(alt: int, of, isp, T_W_ratio: float = 6, m0: float = 68.04):
    T_sl = T_W_ratio * m0 * 9.81

    print(f"SL Thrust due to T/W ratio: \n"
          f"{T_sl:.2f} N\n"
          f"{T_sl*0.224809:.2f} lbs"
          )

    mdot = T_sl / (isp * 9.81)
    print(f"Required mass flow rate: \n"
          f"{mdot:.2f} kg/s\n"
          f"{mdot * 2.20462:.2f} lb/s\n")

    mdot_fuel = None



if __name__ == "__main__":
    # isp_getter(fuel="Kerosene", ox="O2")
    isp, of = isp_getter(fuel="Kerosene", ox="O2")
    rocket_eqn_analysis(alt=1, isp=isp, of=of)