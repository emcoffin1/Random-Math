from HeatTransfer.bartz_formulas import bartz_heat_transfer_const
from MachSolver import mach_from_area_ratio as mach_eps
import numpy as np
from NozzleDesign import build_nozzle
import _extra_utils as utils
from GasProperties import HotGas_Properties, Fluid_Properties

def data_at_point(A, B, value):
    """
    Determine a specific value
    :param A: get index of this
    :param B: and find that index here
    :param value: return value at B
    :return:
    """
    idx = np.argmin(np.abs(A - value))
    return B[idx]

def throat_radius(flow: dict):
    mdot, Pc, gamma, R, T = flow["E"]["mdot"], flow["E"]["Pc"], flow["gamma"], flow["R"], flow["Tc"]
    throat_A = mdot / (Pc * np.sqrt(gamma/(R*T)) * (2/(gamma+1))**((gamma+1)/(2 * (gamma-1))))
    Rt = np.sqrt(throat_A / np.pi)
    return Rt

def isentropic_nozzle_flow(eps, data: dict):
    # Unpack data dictionary
    T0, P0, gamma, R = data["E"]['Tc'], data["E"]['Pc'], data["H"]['gamma'], data["H"]['R']

    # Compute mach through geometry
    M = np.array([mach_eps(eps=e, gamma=gamma) for e in eps])
    # Compute static temp
    T = T0 / (1 + (gamma-1)/2 * M**2)
    # Compute static pressure
    P = P0 * (T/T0)**(gamma/(gamma-1))
    # Compute speed of sound at each station
    a = np.sqrt(gamma * R * T)
    # Compute speed of gas
    U = M * a
    # Compute density
    rho = P / (R * T)

    return {"M": M, "U": U, "T": T, "P": P, "rho": rho}


def main_basic(data: dict):


    # Build nozzle
    x, y, a = build_nozzle(data=data)

    Pe, Pc, Tc, gamma, size, R, k, mu, mdot = (data["E"]["Pe"], data["E"]["Pc"], data["E"]["Tc"], data["H"]["gamma"], data["E"]["size"],
                                      data["H"]["R"], data["H"]["k"], data["H"]["mu"], data["E"]["mdot"])

    # Convert to ratio
    a_min = min(a)
    eps: list = a/a_min


    # Isolate subsonic and supersonic with a negative sign (will be zeroed out eventually)
    ind = np.where(eps == 1.0)[0][0]
    eps[:ind] *= -1

    # == ISENTROPIC FLOW CALCULATIONS == #
    flow: dict = isentropic_nozzle_flow(eps=eps, data=data)
    data["Flow"] = flow

    exit_vel = data["Flow"]["U"][-1]

    mdot_isen = a_min * Pc / np.sqrt(Tc) * np.sqrt(gamma/R*((2/(gamma+1))**((gamma+1)/(gamma-1))))

    print("="*30)
    print("GAS CONDITIONS")
    print(f"Chamber Pressure: {(Pc/1e6):.3f} MPa")
    print(f"Chamber Temperature: {Tc:.2f} K")
    print(f"Gamma: {gamma:.2f}")
    print(f"Gas Constant (R): {R:.2f} J/kg-K")
    print(f"Gas Coefficient of Constant Pressure (cp_g): {data["H"]["cp"]:.2f} J/kg-K")
    print(f"OF Ratio: {data["E"]["OF"]:.2f}")
    print(f"Mass Flow Rate: {mdot:.2f} kg/s")
    of = data["E"]["OF"]
    print(f"Fuel Flow Rate: {(mdot/ (of + 1)):.2f} kg/s")
    print(f"Ox Flow Rate: {(of * mdot / (of + 1)):.2f} kg/s")
    k_gas = data["H"]["k"]
    print(f"Thermal Conductivity: [{k_gas[0]:.2f} -- {k_gas[1]:.2f} -- {k_gas[2]:.2f}] W/(m-K)")
    mu = data["H"]["mu"]
    print(f"Dynamic Viscosity: [{mu[0]:.2f} -- {mu[1]:.2f} -- {mu[2]:.2f}] Pa-s")
    Pr = data["H"]["Pr"]
    print(f"Prandtl Number: [{Pr[0]:.2f} -- {Pr[1]:.2f} -- {Pr[2]:.2f}]")
    print(f"Molar Weight: {data["H"]["MW"]:.2f}")


    print("=" * 30)
    print("ENGINE GEOMETRY")
    print(f"Throat Diameter = {(min(y)*2):.4f} m")
    print(f"Exit Velocity = {exit_vel:.2f} m/s")
    print(f"Exit Diameter = {(y[-1]*2):.2f} m")
    print(f"Total Force @ SL = {(exit_vel * mdot/1e3):.2f} kN")
    print(f"Total Engine Length = {x[-1]:.2f} m")
    print(f"Mass Flow Rate = {mdot_isen:.4f} kg/s")
    print(f"Expansion Ratio = {data["E"]["eps"]:.2f}")

    print("=" * 30)

    # Flow plotting
    flows = [flow["M"], flow["U"], flow["T"], flow["P"], flow["rho"]]
    names = ["M", "U", "T", "P", "rho"]
    subnames = [None, None, None, None, None]

    # == END == #

    # == HEAT TRANSFER == #
    cp = gamma * R / (gamma - 1)
    q: dict = bartz_heat_transfer_const(x=x, y=y, cp=cp,
                                  T=flow["T"], M=flow["M"], info=data)

    # Heat transfer plotting
    # flows1 = [q["hg"], (q["T_wi"], q["T_wo"]), q["qdot"]]
    flows1 = [q["hg"], q["T_wi"], q["qdot"]]
    names1 = ["Heat Transfer Coefficient", "Init Wall Temps", "Heat Transfer Rate"]
    # subnames1 = [None, ["Inner Temps", "Outer Temps"], None]
    subnames1 = [None, None, None]

    max_wall_temp_x = data_at_point(A=q["T_wi"], B=x, value=np.max(q["T_wi"]))

    print("HEAT DATA")
    print(f"Max Wall Temp = {np.max(q["T_wi"]):.2f} K at {(max_wall_temp_x*1000):.2f} mm from throat")
    print(f"Average Heat Transfer Coefficient (hg) = {np.mean(q['hg']):.2f} W/m^2-k")
    print(f"Maximum Heat Transfer Coefficient (hg) = {max(q['hg']):.2f} W/m^2-k")
    print(f"Average Heat Flux (q')= {np.mean(q['qdot'])/1e6:.2f} MW/m^2")
    print(f"Maximum Heat Flux (q') = {max(q['qdot'])/1e6:.2f} MW/m^2")

    # == END == #

    # Tc_out is the temperature of coolant leaving the regens and entering the injector
    # Tc_out is also a target value
    # Compute the total heat transfer and mdot
    Tc_out = 350
    # Q: dict = total_heat_flux(qdot=q["qdot"], x=x, y=y, cp=cp, Tc_in=q["T_ci"], Tc_out=Tc_out)
    # # print(f"Mass flow rate (first pass): {Q["mdot"]:.3f}kg/s")
    # # print(f"Total heat flux (Q): {Q["Qtotal"]:.2f}W")
    # flows2 = [Q["Q"]]
    # names2 = ["Total Q"]
    # subnames2 = [None]


    # == PLOTTING == #
    flows = flows + flows1 #+ flows2

    names = names + names1 #+ names2
    subnames = subnames + subnames1 #+ subnames2
    utils.plot_flow_chart(x=x, data=flows, labels=names, sublabels=subnames)

    utils.plot_flow_field(x, y, data=q["qdot"], label="Heat Flux")


if __name__ == '__main__':
    """Label CEA as true if you want to use CEA as gas properties"""
    """
    All variables in equations are assumed to be in reference to the hot gas unless denoted otherwise
        ie  coolant values   : cp_c 
            lox values       : cp_l
            fuel values      : cp_f 
    """

    # == ENGINE INFO == #

    # info = {"CEA": True,
    #         "Pc": 2.013e6,       # Chamber Pressure [Pa]
    #         "Pe": 101325,    # Ambient Pressure (exit) [Pa]
    #         "Tc": 3500,         # Chamber temp [K]
    #         "mdot": 1.89,       # Mass Flow Rate [kg/s]
    #         "gamma": 1.4,      # Hot gas specific heat ratio
    #         "R": 287,           # Hot gas ideal gas constant
    #         "mu": 8.617e-4,     # Hot gas dynamic viscosity
    #         "k": 16.27,        # Wall thermal conductivity
    #         "size": 1.0,        # Engine proportion (% of rao nozzle)
    #         "plots": "no",       # Choice of engine plotting (no, 2D, 3D)
    #         "Fuel": "RP-1",
    #         "Ox": "LOX",
    #         "OF": 1.8,
    #         "eps": None,
    #         "cp_g": None,
    #         "cstar": None,
    #         "MW": None
    #         }
    #
    info = {"CEA": True,
            "plots": "no",
            "E": {
                "Pc": 2.013e6,  # Chamber Pressure [Pa]
                "Pe": 101325,  # Ambient Pressure (exit) [Pa]
                "Tc": 3500,  # Chamber temp [K]
                "mdot": 1.89,  # Mass Flow Rate [kg/s]
                "OF": 1.8,
                "size": 1.0,
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
                "T": 298,
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
            }
            }

    if info["CEA"]:
        # Run rocketcea
        HotGas_Properties(Pc=info["E"]["Pc"], fuel=info["F"]["Type"], ox=info["O"]["Type"], OF=info["E"]["OF"], dic=info)
        Fluid_Properties(dic=info)
        print(f"FUEL: {info["F"]}")
        print(f"LOX: {info["O"]}")

    # print(f"run 1: {info}")
    # == RUN ME == #
    main_basic(data=info)
