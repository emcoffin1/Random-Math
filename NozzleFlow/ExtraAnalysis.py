from rocketcea.blends import newOxWithNewState

from NozzleDesign import build_nozzle
import numpy as np
import matplotlib.pyplot as plt
from MachSolver import isentropic_nozzle_flow
from GasProperties import HotGas_Properties
import copy

def separation_location(data:dict, c_sep: float=0.4):
    Pa = data["E"]["Pe"]
    P = data["Flow"]["P"]
    x = data["E"]["x"]
    thresh = c_sep * Pa

    idx = np.where(P <= thresh)[0]
    if len(idx) == 0:
        return None

    if x[idx[0]] <= 0:
        return None

    return float(x[idx[0]])



def Startup_Analysis(data:dict):

    x = data["E"]["x"]
    if x is None:
        x, y, a = build_nozzle(data=data)
    else:
        y = data["E"]["y"]
        a = data["E"]["a"]

    t_ign               = 0
    t_ign_delay         = 0.05    # s
    tau                 = 0.1             # s (50-200)
    dt                  = 0.002            # s
    t_end               = 5 * tau

    A_spike             = 0.1
    tau_spike           = 0.02

    Pc_steady           = data["E"]["Pc"]
    t0                  = t_ign + t_ign_delay

    t                   = np.arange(0, t_end+dt, dt)
    Pc                  = np.zeros_like(t)

    gamma               = data["H"]["gamma"]

    for i, ti in enumerate(t):
        if ti >= t0:
            Pc[i]       = Pc_steady * (1 - np.exp(-(ti-t0)/tau))
            Pc[i]       *= (1 + A_spike * np.exp(-(ti-t0)/tau_spike))

    # print(f"Maximum Chamber Pressure: {np.max(Pc)}")
    # plt.plot(t, Pc)
    # plt.show()

    # NPR = Pc(t) / Pa
    # ==================================== #
    # == NOZZLE PRESSURE RATIO ANALYSIS == #
    # ==================================== #
    P_ambient           = data["E"]["Pe"]
    NPR                 = Pc / P_ambient

    NPR_ideal           = Pc_steady / P_ambient
    NPR_critical        = ((gamma+1)/2)**(gamma/(gamma-1))

    # plt.plot(t, NPR)
    # plt.axhline(y=NPR_critical, color="b")
    # plt.show()

    # =========================================== #
    # == COMPUTE ISENTROPIC FLOW AT EACH POINT == #
    # =========================================== #
    eps                 = data["E"]["aspect_ratio"]
    ind                 = np.where(eps == 1.0)[0][0]
    eps[:ind]           *= -1

    isen_flows = []
    separation_points = []
    for i in range(len(t)):
        data_c = copy.deepcopy(data)
        data_c["E"]["Pc"] = Pc[i]
        # HotGas_Properties(dic=data_c, forced=False)
        isentropic_nozzle_flow(data=data_c, eps=eps)
        isen_flows.append(data_c)

        sep = separation_location(data=data_c)
        separation_points.append(sep)

        progress = i / (len(t) - 1) * 100
        print(f"\rIsentropic solver progress: {progress:5.1f}%", end="", flush=True)

    print(f"\nIsentropic Flow and separation points solved for!")
    # PLOT
    # plt.plot(t, separation_points)
    # plt.ylabel("Separation Points")
    # plt.xlabel("Time")
    # plt.show()

    # ======================= #
    # == RECOVERY PRESSURE == #
    # ======================= #
    L_recov = 0.1 * (x[-1] - x[0])
    for i, snap in enumerate(isen_flows):
        sep_x = separation_points[i]
        if sep_x is None:
            continue

        x_arr = snap["E"]["x"]
        P_old = snap["Flow"]["P"]

        idx_sep = np.argmin(np.abs(x_arr - sep_x))
        P_sep = P_old[idx_sep]

        for j in range(len(x_arr)):
            if j < idx_sep:
                continue

            dx = x_arr[j] - x_arr[idx_sep]
            P_old[j] = P_ambient + (P_sep - P_ambient) * np.exp(-dx / L_recov)

        snap["Flow"]["P"] = P_old


    # index = 20
    # for _ in range(15):
    #     if separation_points[index] is None:
    #         index += 1
    #     else:
    #         break
    # print(f"Chamber Pressure at analyzed point: {isen_flows[index]["E"]["Pc"]}")
    # plt.plot(isen_flows[index]["E"]["x"], isen_flows[index]["Flow"]["P"])
    # plt.axvline(x=separation_points[index], color="r", linestyle="--")
    # plt.show()
    #
    # plt.plot(x, isen_flows[index]["Flow"]["P"], label="early")
    # plt.plot(x, isen_flows[50]["Flow"]["P"], label="mid")
    # plt.plot(x, isen_flows[-1]["Flow"]["P"], label="late")
    # plt.legend()
    # plt.show()





