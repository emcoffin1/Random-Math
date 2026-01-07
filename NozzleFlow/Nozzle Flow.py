from copy import deepcopy

from GeometryDesign import *
from HeatTransfer.bartz_formulas import bartz_heat_transfer_const, bartz_heat_transfer_1d
from MachSolver import isentropic_nozzle_flow
import numpy as np
from NozzleDesign import build_nozzle
import _extra_utils as utils
from GasProperties import HotGas_Properties, Fluid_Properties, Material_Properties
import matplotlib.pyplot as plt
import copy
from ExtraAnalysis import *
from NozzleFlow.GasProperties import init_cea


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


def iterate_cooling_design(q: dict, data: dict, tol=0.05, iter=50, alpha=0.2):

    x = data["E"]["x"]
    Tb = data_at_point(A=x, B=q["T_cool"], value=0)

    for _ in range(iter):

        data["F"]["T"] = 298
        Fluid_Properties(dic=data, coolant_only=True)

        CoolantSizingGuide(data=data, coolant_bulk_temp=Tb, display=False)
        q = bartz_heat_transfer_1d(info=data)
        Tb1 = data_at_point(A=x, B=q["T_cool"], value=0)

        if np.abs(Tb1 - Tb) < tol:
            CoolantSizingGuide(data=data, coolant_bulk_temp=Tb, display=True)
            return q

        Tb = Tb1


def main_basic(data: dict, nozzle_build: bool = True, display=True):
    frmt                = "{:<50} {:<10.3f} {:<10} {:<}"
    frmt2               = "{:<50} {:<10} {:<10} {:<}"
    frmte               = "{:<50} {:<10.3e} {:<10} {:<}"

    iterate_cooling     = data["iterate_cooling"]
    channel_plot        = data["ChannelPlot"]

    if nozzle_build:
        # Build nozzle
        x, y, a         = build_nozzle(data=data)
        if iterate_cooling:
            CoolantSizingGuide(data=data, display=False, coolant_bulk_temp=350)
        else:
            CoolantSizingGuide(data=data, display=True)

        non_throat_cooling_geom(data=data)

    else:
        x = 1
        y = 2
        a = 3

    Pe, Pc, Tc, gamma, size, R, k, mu, mdot = (data["E"]["Pe"], data["E"]["Pc"], data["E"]["Tc"], data["H"]["gamma"],
                                               data["E"]["size"], data["H"]["R"], data["H"]["k"], data["H"]["mu"],
                                               data["E"]["mdot"])

    eps                 = data["E"]["aspect_ratio"]
    a_min               = np.min(a)

    # Isolate subsonic and supersonic with a negative sign (will be zeroed out eventually)
    ind                 = np.where(eps == 1.0)[0][0]
    eps[:ind]           *= -1
    # print(1)
    # CoolantSizingGuide(data=data, display=True)
    # print(2)
    # == ISENTROPIC FLOW CALCULATIONS == #
    isentropic_nozzle_flow(eps=eps, data=data)
    # film_cooling(data=data)
    # return

    exit_vel            = data["Flow"]["U"][-1]
    mdot_isen           = (a_min * Pc / np.sqrt(Tc) * np.sqrt(gamma/R*((2/(gamma+1))**((gamma+1)/(gamma-1)))))

    eps                 = abs(eps)
    data["E"]["aspect_ratio"] = eps

    # Flow plotting
    flows               = [data["Flow"]["M"], data["Flow"]["U"], data["Flow"]["T"], data["Flow"]["P"], data["Flow"]["rho"],]
    names               = ["M", "U", "T", "P", "rho"]
    subnames            = [None, None, None, None, None]

    # == END == #

    # =================== #
    # == HEAT TRANSFER == #
    # =================== #

    analyze             = True
    dims                = data["dimensions"]
    if dims == 0:
        q: dict         = bartz_heat_transfer_const(info=data)
    elif dims == 1:

        q: dict         = bartz_heat_transfer_1d(info=data)

        if iterate_cooling:
            q           = iterate_cooling_design(q=q, data=data, tol=1e-4)

        data["q"]       = q

        # Generate complete cooling geometry
        non_throat_cooling_geom(data=data)
        # Display the geom
        if channel_plot:
            view_channel_slices(data=data)

    else:
        analyze         = False
        q               = dict()

    # =========================== #
    # == Perform Wall Analysis == #
    # =========================== #



    # Not display used for external iterations
    if not display:
        return np.max(q["T_cool"]), np.max(q["T_wall_gas"]), np.max(data["W"]["yield_strength"]/(data["Flow"]["P"]*data["E"]["y"]/data["W"]["thickness"]))
    else:
        # Analyze is used to view the heat transfer information
        if analyze:
            # Heat transfer plotting
            flows1          = [(q["h_hg"], q["h_wc"]),
                               (q["T_wall_gas"], q["T_wall_coolant"], q["T_aw"]),
                               (q["R_hg_w"], q["R_w_w"], q["R_w_c"]),
                               q["Q_dot"],
                               q["Re"],
                               q["T_cool"]
                               ]

            names1          = ["Heat Transfer Coefficients",
                               "Wall Temps",
                               "Wall Resistances",
                               "Q_dot",
                               "Reynolds",
                               "Coolant Temp"
                               ]

            subnames1       = [("Gas-Wall", "Wall-Coolant"),
                               ("Wall-Gas", "Wall-Coolant", "Adiabatic Wall"),
                               ("Wall-Gas", "Wall-Wall", "Wall-Coolant"),
                               None,
                               None,
                               None
                               ]

            max_wall_temp_x = data_at_point(A=q["T_wall_gas"], B=x, value=np.max(q["T_wall_gas"]))
            max_wall_temp   = np.max(q["T_wall_gas"])

        Lc = data["E"]["Lc"] if data["E"]["Lc"] is not None else 0

        # ============== #
        # == PRINTING == #
        # ============== #
        print("=" * 72, f"{'|':<}")
        print(f"{'ENGINE GEOMETRY':^70} {'|':>3}")
        print("- " * 36, f"{'|':<}")

        print(frmt.format("Throat Diameter", min(y) * 2*100, "cm", "|"))
        print(frmt.format("Exit Velocity", exit_vel, "m/s", "|"))
        print(frmt.format("Exit Diameter", y[-1] * 2, "m", "|"))
        print(frmt.format("Total Force @ SL", exit_vel * mdot / 1e3, "kN", "|"))
        print(frmt.format("Total Engine Length", x[-1] - x[0], "m", "|"))
        print(frmt.format("Mass Flow Rate", mdot_isen, "kg/s", "|"))
        print(frmt.format("Expansion Ratio", np.max(eps), "kg/s", "|"))
        print(frmt.format("Chamber Radius", data["E"]["y"][-1]*1000, "mm", "|"))
        print(frmt.format("Engine Length", (x[-1] - x[0]), "m", "|"))
        print(frmt.format("Chamber Length", Lc*1000, "mm", "|"))
        print(frmt.format("Characteristic Velocity", data["H"]["cstar"], "m/s", "|"))

        print("="*72,f"{'|':<}")
        print(f"{'GAS CONDITIONS':^70} {'|':>3}")
        print("- " * 36, f"{'|':<}")

        P_exit = data["Flow"]["P"][-1]
        P_ambient = data["E"]["Pe"]
        if P_exit < P_ambient:
            condition = "Over"
        elif P_exit > P_ambient:
            condition = "Under"
        else:
            condition = "Perfect"

        print(frmt.format("Chamber Pressure", Pc/1e6, "MPa", "|"))
        print(frmt.format("Chamber Temperature", Tc, "K", "|"))
        print(frmt.format("Exit Pressure", data["Flow"]["P"][-1] / 1e6, "MPa", "|"))
        print(frmt2.format("Expansion Condition", condition, "", "|"))
        print(frmt.format("Ambient Pressure", P_ambient / 1e6, "MPa", "|"))

        print(frmt.format("Gamma", gamma, "", "|"))
        print(frmt.format("Gas Constant (R)", R, "J/kg-K", "|"))
        print(frmt.format("Gas Coefficient of Constant Pressure (cp_g)", data["H"]["cp"][1], "J/kg-K", "|"))
        print(frmt.format("OF Ratio", data["E"]["OF"], "", "|"))
        print(frmt.format("Mass Flow Rate", mdot, "kg/s", "|"))
        of                  = data["E"]["OF"]
        print(frmt.format("Fuel Flow Rate", mdot/ (of + 1), "kg/s", "|"))
        print(frmt.format("Ox Flow Rate", of * mdot / (of + 1), "kg/s", "|"))
        k_gas               = data["H"]["k"]
        print(frmt2.format("Thermal Conductivity", "", "", "|"))
        print(frmt.format(" ", k_gas[0], "W/(m-K)", "|"))
        print(frmt.format(" ", k_gas[1], "W/(m-K)", "|"))
        print(frmt.format(" ", k_gas[2], "W/(m-K)", "|"))
        mu                  = data["H"]["mu"]
        print(frmt2.format("Dynamic Viscosity", "", "", "|"))
        print(frmte.format("", mu[0], "Pa-s", "|"))
        print(frmte.format("", mu[1], "Pa-s", "|"))
        print(frmte.format("", mu[2], "Pa-s", "|"))
        Pr                  = data["H"]["Pr"]
        print(frmt2.format("Prandtl Number", "", "", "|"))
        print(frmt.format("", Pr[0], "", "|"))
        print(frmt.format("", Pr[1], "", "|"))
        print(frmt.format("", Pr[2], "", "|"))
        print(frmt.format("Molar Weight", data["H"]["MW"], "g/mol", "|"))


        if analyze:
            print("="*72,f"{'|':<}")
            print(f"{'HEAT DATA':^70} {'|':>3}")
            print("- " * 36, f"{'|':<}")

            print(frmt.format("Maximum Wall Temp", max_wall_temp, "K", "|"))
            print(frmt.format("at ... from throat", max_wall_temp_x*1000, "mm", "|"))
            print(frmt.format("Maximum Coolant Temp", np.max(q["T_cool"]), "K", "|"))
            print(frmt.format("Coolant Temp at Throat", data_at_point(A=data["E"]["x"], B=q["T_cool"], value=0), "K", "|"))
            # if np.max(q["T_cool"]) == data["F"]["T_max"]:
            #     print(frmt2.format("The coolant exceeded the thermally stable temperature region", "","","|"))
            #     print(frmt.format("The coolant was therefor clamped to", data["F"]["T_max"], "K", "|"))
            print(frmt.format("Average Heat Transfer Coefficient (hot gas)", np.mean(q["h_hg"])/1000, "kW/m^2-K", "|"))
            print(frmt.format("Maximum Heat Transfer Coefficient (hot gas)", max(q["h_hg"])/1000, "kW/m^2-K", "|"))
            print(frmt.format("Maximum Heat Transfer Coefficient (wall->coolant", max(q["h_wc"])/1000, "kW/m^2-K", "|"))
            print(frmt.format("Average Heat Rate (Qdot)", np.mean(q["Q_dot"]), "W", "|"))
            print(frmt.format("Total Heat rate (Qdot)", sum(q["Q_dot"]), "W", "|"))

            melting_point   = data["W"]["solidus"]
            if max_wall_temp > melting_point:
                excess      = melting_point - max_wall_temp
                print(f"WARNING : Maximum wall temp exceeds the melting point of {data["W"]["Type"]} by {abs(excess):.2f} K")
                percent     = abs(melting_point - max_wall_temp) / max_wall_temp * 100
                print(f"WARNING : This is a {percent:.2f}% error")

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
        if analyze:
            flows           = flows1 + flows
            names           = names1 + names
            subnames        = subnames1 + subnames

        else:
            flows           = flows
            names           = names
            subnames        = subnames
        utils.plot_flow_chart(x=x, data=flows, labels=names, sublabels=subnames)

        utils.plot_flow_field(x, y, data=q["T_wall_gas"], label="Inner Wall Temperature")



if __name__ == '__main__':
    """Label CEA as true if you want to use CEA as gas properties"""
    """
    All variables in equations are assumed to be in reference to the hot gas unless denoted otherwise
        ie  coolant values   : cp_c 
            lox values       : cp_l
            fuel values      : cp_f 
    """

    info = {"CEA": True,
            "plots": "no",
            "dimensions": 1,    # Complexity of heat transfer
            "iterate_cooling": True,
            "FilmCool": True,
            "ChannelPlot": False,
            "CEA_obj": object,
            "rp1_prop_obj": object,
            "E": {
                "Pc": 6.013e6,  # Chamber Pressure [Pa]
                "Pe": 101325,  # Ambient Pressure (exit) [Pa]
                "Tc": 3600,  # Chamber temp [K]
                "mdot": 1.89,  # Mass Flow Rate [kg/s]
                "OF": 1.8,
                "size": 0.8,
                "CR": 7,
                "Lc": None,
                "x": None,
                "y": None,
                "a": None,
                "aspect_ratio": None,
            },
            "H": {
                "mu": None,
                "k": None,
                "rho": None,
                "gamma": None,
                "cp": None,
                "cstar": None,
                "MW": None,
                "R": None,
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
            },
            "W": {
                # "Type": "SS 316L",
                # "Type": "Tungsten",
                # "Type": "Copper Chromium",
                "Type": "Inconel 718",
                "thickness": 0.00025
            },
            "C": {
                "Type": "Square",
                "pass": 1,           # Number of passes of each channel standard = 1
                "spacing": 0.000824,   # Fin thickness -- space between channels
                "height": 0.000824,     # Channel height
                "num_ch": 55,
                "h": None,
                "Nu": None,
                "Re": None,
                "wall_thickness": 0.00025,
                "depth_arr": None,
                "width_arr": None,
            },
            "Flow": {
                "P": None,
            },
            "Injector": {
                "dP": None,
                "tau_eff": None,
                "L_fuel": 0.1,
                "L_ox": 0.2,

            }


            }
    # == ENGINE INFO == #


    if info["CEA"]:
        # Run rocketcea
        init_cea(data=info)
        HotGas_Properties(dic=info)
        Fluid_Properties(dic=info)
        Material_Properties(dic=info)
    #
    # # print(f"run 1: {info}")
    # == RUN ME == #
    main_basic(data=info, nozzle_build=True)
    # Startup_Analysis(data=info)
    # First_Modal_Analysis(data=info)
    # CoolantSizingGuide(data=info)
    # film_cooling(data=info)


    # l = np.linspace(0.01,0.5, 50)
    # results = []
    # fos_arr = []
    # t_wall_arr = []
    # for i in l:
    #
    #     info = {"CEA": True,
    #         "plots": "no",
    #         "dimensions": 1,    # Complexity of heat transfer
    #         "E": {
    #             "Pc": 3e6,  # Chamber Pressure [Pa]
    #             "Pe": 101325,  # Ambient Pressure (exit) [Pa]
    #             "Tc": 3500,  # Chamber temp [K]
    #             "mdot": 0.73,  # Mass Flow Rate [kg/s]
    #             "OF": 2.25,
    #             "size": 1.0,
    #             "CR": 8,
    #             "Lc": None,
    #             "x": None,
    #             "y": None,
    #         },
    #         "H": {
    #             "mu": None,
    #             "k": None,
    #             "rho": None,
    #             "gamma": None,
    #             "cp": None,
    #             "cstar": None,
    #             "MW": None,
    #         },
    #         "F": {
    #             "Type": "RP-1",
    #             "T": 298,
    #             "P": None,
    #             "mu": None,
    #             "k": None,
    #             "rho": None,
    #             "gamma": None,
    #             "cp": None,
    #             "cstar": None,
    #             "MW": None,
    #             "mdot": None,
    #         },
    #         "O": {
    #             "Type": "LOX",
    #             "T": 98,
    #             "P": None,
    #             "mu": None,
    #             "k": None,
    #             "rho": None,
    #             "gamma": None,
    #             "cp": None,
    #             "cstar": None,
    #             "MW": None,
    #             "mdot": None,
    #         },
    #         "W": {
    #             # "Type": "SS 316L",
    #             # "Type": "Tungsten",
    #             # "Type": "Copper Chromium",
    #             "Type": "Inconel 718",
    #             "thickness": i
    #         },
    #         "C": {
    #             "Type": "Square",
    #             "spacing": 0.0006,   # Fin thickness -- space between channels
    #             "height": 0.0006,     # Channel height
    #             "num_ch": 120,
    #             "h": None,
    #             "Nu": None,
    #             "Re": None,
    #         },
    #         "Flow": {
    #             "x": None,
    #             "y": None,
    #             "a": None,
    #             "eps": None
    #         },
    #
    #
    #         }
    #
    #
    #
    #     if info["CEA"]:
    #         # Run rocketcea
    #         HotGas_Properties(Pc=info["E"]["Pc"], fuel=info["F"]["Type"], ox=info["O"]["Type"], OF=info["E"]["OF"],
    #                           dic=info)
    #         Fluid_Properties(dic=info)
    #         Material_Properties(dic=info)
    #     h_max, t_wall, fos = main_basic(data=info, display=False, nozzle_build=True)
    #     results.append(h_max)
    #     fos_arr.append(fos)
    #     t_wall_arr.append(t_wall)
    #
    # # for i in range(len(l)):
    # #     print(f"{l[i]:.2f} m: {results[i]:.2f} K   ===   {fos_arr[i]:.2f} Pa")
    #
    # plt.plot(l, np.gradient(l, results))
    # # plt.axhline(y=588, linestyle="--", color="orange")
    # plt.show()
    #
    # plt.plot(l, np.gradient(l,t_wall_arr))
    # plt.show()


