import numpy as np
import matplotlib.pyplot as plt
from HeatTransfer.bartz_formulas import bartz_heat_transfer_1d, pressure_drop_assessment
import _extra_utils as utils
from GasProperties import HotGas_Properties, Fluid_Properties, Material_Properties
from ExtraAnalysis import *
from NozzleFlow.GasProperties import init_cea


def throat_radius(flow: dict):
    mdot, Pc, gamma, R, T = flow["E"]["mdot"], flow["E"]["Pc"], flow["gamma"], flow["R"], flow["Tc"]
    throat_A = mdot / (Pc * np.sqrt(gamma/(R*T)) * (2/(gamma+1))**((gamma+1)/(2 * (gamma-1))))
    Rt = np.sqrt(throat_A / np.pi)
    return Rt


def iterate_cooling_design(q: dict, data: dict, tol=0.05, iter=50, alpha=0.2):

    x = data["E"]["x"]
    Tb = utils.data_at_point(A=x, B=q["T_cool"], value=0)

    for _ in range(iter):

        # Reset the coolant inlet values
        data["F"]["T"] = 298
        data["F"]["H"] = 0
        Fluid_Properties(dic=data, coolant_only=True)

        CoolantSizingGuide(data=data, coolant_bulk_temp=Tb, display=False)
        non_throat_cooling_geom(data=data)
        pressure_drop_assessment(data=data)

        q = bartz_heat_transfer_1d(info=data)
        Tb1 = utils.data_at_point(A=x, B=q["T_cool"], value=0)

        if np.abs(Tb1 - Tb) < tol:
            CoolantSizingGuide(data=data, coolant_bulk_temp=Tb, display=False)
            non_throat_cooling_geom(data=data)
            pressure_drop_assessment(data=data)
            return q

        Tb = Tb1

    return q


def iterate_pressure_drop(data, tol=0.05, iter=50):
    P0 = data["C"]["dP"]
    for i in range(iter):

        data["F"]["T"] = 298
        data["F"]["H"] = 0
        Fluid_Properties(dic=data, coolant_only=True)

        pressure_drop_assessment(data=data)
        _ = bartz_heat_transfer_1d(info=data)

        P1 = data["C"]["dP"]
        dif = np.abs(P1 - P0)

        if dif < tol:
            return
        P0 = P1

    raise OverflowError(f"FAILED TO CONVERGE PRESSURE DROP TEST")

def iterate_using_pressure_drop(data, q, tol=0.01, iter=50):
    dP_old = q["dP_arr"]
    q_new = None
    for i in range(iter):
        data["F"]["StartingPressure"] = data["Injector"]["dP"] + q["P_c"][-1]
        q_new = bartz_heat_transfer_1d(info=data)
        dP_new = q_new["dP_arr"]

        dif = abs(dP_new - dP_old) / dP_old

        if dif < tol:
            return q_new
        dP_old = dP_new
        q = q_new
    print(f"FAILED TO CONVERGE PRESSURE DROP TEST")
    return q_new







def main_basic(data: dict, nozzle_build: bool = True, display=True):

    # ===================== #
    # == SOLVER SETTINGS == #
    # ===================== #

    iterate_cooling     = data["Solver"]["IterateCooling"]
    iterate_pressure    = data["Solver"]["IteratePressureDrop"]
    iterate_using_pressure = data["Solver"]["IterateUsingPressureDrop"]
    heat_solver         = data["Solver"]["HeatSolver"]

    print_display       = data["Display"]["PrintOut"]
    flow_plot           = data["Display"]["FlowPlot"]
    energy_plot         = data["Display"]["EnergyPlot"]
    contour_plot        = data["Display"]["ContourPlot"]

    # ============================ #
    # == FORMULATE THE GEOMETRY == #
    # ============================ #

    if nozzle_build:
        # Build nozzle
        x, y, a         = build_nozzle(data=data)
        iterate = False if iterate_cooling else True

        # First pass assessment
        # CoolantSizingGuide(data=data, display=iterate)
        CoolantSizingGuide(data=data)
        non_throat_cooling_geom(data=data)
        pressure_drop_assessment(data=data)

    else:
        x, y, a = 1, 2, 3

    eps                 = data["E"]["aspect_ratio"]

    # Isolate subsonic and supersonic with a negative sign (will be zeroed out eventually)
    ind                 = np.where(eps == 1.0)[0][0]
    eps[:ind]           *= -1

    # ================================== #
    # == ISENTROPIC FLOW CALCULATIONS == #
    # ================================== #

    isentropic_nozzle_flow(eps=eps, data=data)
    eps                 = abs(eps)
    data["E"]["aspect_ratio"] = eps

    # == END == #

    # =================== #
    # == HEAT TRANSFER == #
    # =================== #
    if heat_solver:
        # HotGas_Properties(dic=data)
        q: dict         = bartz_heat_transfer_1d(info=data)
        if iterate_using_pressure:
            print("UPDATE: Iterating Using Pressure Drop")
            q           = iterate_using_pressure_drop(data=info, q=q)
        # q: dict = heat_transfer_solver(data=data)
        if iterate_cooling:
            print("UPDATE: Iterating Cooling Design")
            q           = iterate_cooling_design(q=q, data=data, tol=1e-4)

        # Perform an analysis on the pressure drop in the regen channel
        if iterate_pressure:
            print("UPDATE: Iterating Using Pressure Drop")
            iterate_pressure_drop(data=data)

        data["q"]       = q
        # This just updates the channel pressure drop printout
        # with the most up-to-date coolant data
        pressure_drop_assessment(data=data)

    if print_display:
        utils.data_display(data=data)

    if flow_plot or energy_plot or contour_plot:
        utils.plot_info(data=data)


if __name__ == '__main__':
    """Label CEA as true if you want to use CEA as gas properties"""
    """
    All variables in equations are assumed to be in reference to the hot gas unless denoted otherwise
        ie  coolant values   : cp_c 
            lox values       : cp_l
            fuel values      : cp_f 
    """

    info = {
            "Solver": {
                "CEA": True,
                "IterateCooling": False,
                "IteratePressureDrop": False,
                "IterateUsingPressureDrop": False,
                "FilmCool": False,
                "EnergyMethod": True,
                "HeatSolver": True,
            },
            "Display": {
                "PrintOut": True,
                "EnginePlot": "no",
                "FlowPlot": False,
                "EnergyPlot": True,
                "ChannelPlot": False,
                "ContourPlot": False,
            },
            "CEA_obj": object,
            "E": {
                "Pc": 3.01e6,  # Chamber Pressure [Pa]
                "Pe": 101325,  # Ambient Pressure (exit) [Pa]
                "Tc": None,  # Chamber temp [K]
                "mdot": 2.75,  # Mass Flow Rate [kg/s]
                "OF": 1.8,
                "size": 1.0,
                "CR": 6,
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
                "Type": "Kerosene",
                "State": None,
                "T": 298,
                "P": 6e6,
                "H": 0,
                "mu": None,
                "k": None,
                "rho": None,
                "gamma": None,
                "cp": None,
                "cstar": None,
                "MW": None,
                "mdot": None,
                "rho_arr": [],
                "mu_arr": [],
            },
            "O": {
                "Type": "LOX",
                "State": None,
                "T": 98,
                "P": 6e6,
                "H": 0,
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
                # "Type": "Inconel 718",
                # "Type": "GRCop-42",
                "thickness": 0.001,
                "InitialTemp": 298,
                "T": 298,
                "roughness": 1e-5,
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
                "dP_arr": np.ndarray,
            },
            "Flow": {
                "P": None,
            },
            "Injector": {
                "dP": 1.2,
                "tau_eff": None,
                "L_fuel": 0.1,
                "L_ox": 0.2,
            },
            "q": None
            }
    info["Injector"]["dP"] = info["Injector"]["dP"] * info["E"]["Pc"]
    # == ENGINE INFO == #

    if info["Solver"]["CEA"]:
        # Run rocketcea
        init_cea(data=info)
        HotGas_Properties(dic=info)
        Fluid_Properties(dic=info)
        Material_Properties(dic=info)



    # == RUN ME == #
    main_basic(data=info, nozzle_build=True)
    # Startup_Analysis(data=info)
    # First_Modal_Analysis(data=info)
    # CoolantSizingGuide(data=info)
    # film_cooling(data=info)

