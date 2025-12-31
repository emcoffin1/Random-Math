from rocketcea.blends import newOxWithNewState
from GeometryDesign import *
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


def compute_load_eps(sep_points, t, eps_min=0.01, eps_max=0.08):
    """
    Compute separation velocity
    """
    xsep = np.array([np.nan if s is None else s for s in sep_points], dtype=float)

    v = np.zeros_like(xsep)
    dt = t[1] - t[0]

    for i in range(1, len(xsep)):
        if np.isfinite(xsep[i]) and np.isfinite(xsep[i-1]):
            v[i] = (xsep[i-1] - xsep[i]) / dt
        else:
            v[i] = 0.0

    vmax = np.max(np.abs(v)) if np.max(np.abs(v)) > 0.0 else 1.0
    eps = eps_min + (eps_max - eps_min) * (np.abs(v) / vmax)
    return eps, v


def lateral_load_per_snap(snap, sep_x, load_eps, x_ref):
    x = snap["E"]["x"]
    y = snap["E"]["y"]
    Pa = snap["E"]["Pe"]
    P = snap["Flow"]["P"] - Pa

    if sep_x is None or load_eps == 0.0:
        return 0.0, 0.0

    i_sep = int(np.argmin(np.abs(x - sep_x)))

    # Integrate using trapezoid method
    mask = np.arange(len(x)) >= i_sep
    Fx_prime = P[mask] * y[mask]
    Mx_prime = P[mask] * y[mask] * (x[mask] - x_ref)
    Fx = np.pi * load_eps * np.trapezoid(Fx_prime, x[mask])
    Mx = np.pi * load_eps * np.trapezoid(Mx_prime, x[mask])

    return Fx, Mx

def compute_load_stress(snap, M, F, x_ref):

    t_wall = snap["W"]["thickness"]
    height_channel = snap["C"]["height"]

    t_min = float(snap.get("t_min", 0.0005))
    # effective wall thickness condisdering channel geometry
    t_eff = np.maximum(t_min, t_wall - height_channel)

    P = snap["Flow"]["P"]
    y = snap["E"]["y"]
    x = snap["E"]["x"]

    # median radius (using radius to wall and effective wall thickness)
    r_m = y + (0.5 * t_eff)

    hoop_stress = P * y / t_eff
    long_stress = hoop_stress / 2


    # moment of inertia, assumed thin circular ring
    # uses mean radius and wall thickness
    I = np.pi * (r_m**3) * t_eff

    bend_moment = M + F * (x - x_ref)
    bend_stress = bend_moment * r_m / I

    long_stress_plus = long_stress + bend_stress
    long_stress_minus = long_stress - bend_stress

    von_stress_plus = np.sqrt(hoop_stress ** 2 + long_stress_plus ** 2 - hoop_stress*long_stress_plus)
    von_stress_minus = np.sqrt(hoop_stress**2 + long_stress_minus**2 - hoop_stress*long_stress_minus)

    return np.maximum(von_stress_plus, von_stress_minus)



def Startup_Analysis(data:dict):

    x = data["E"]["x"]
    if x is None:
        x, y, a = build_nozzle(data=data)
        cooling_geometry(dic=data)

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


    index = 20
    for _ in range(15):
        if separation_points[index] is None:
            index += 1
        else:
            break
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

    # ======================= #
    # == LOAD CALCULATIONS == #
    # ======================= #

    load_eps_t, vsep = compute_load_eps(sep_points=separation_points, t=t)

    x_ref = x[np.argmin(isen_flows[0]["E"]["x"])]   # just the throat point

    F_lat = np.zeros(len(isen_flows))
    M_lat = np.zeros(len(isen_flows))

    for i, snap in enumerate(isen_flows):
        F_lat[i], M_lat[i] = lateral_load_per_snap(snap=snap,
                                                   sep_x=separation_points[i],
                                                   load_eps=load_eps_t[i],
                                                   x_ref=x_ref)


        snap["E"]["stress"] = compute_load_stress(snap=snap, M=M_lat[i], F=F_lat[i], x_ref=x_ref)



    plt.plot(x, isen_flows[index]["E"]["stress"])
    plt.show()





