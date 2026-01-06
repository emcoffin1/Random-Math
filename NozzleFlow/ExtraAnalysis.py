from rocketcea.blends import newOxWithNewState
from GeometryDesign import *
from NozzleDesign import build_nozzle
import numpy as np
import matplotlib.pyplot as plt
from MachSolver import isentropic_nozzle_flow
from GasProperties import HotGas_Properties
import copy

from NozzleFlow.GasProperties import Fluid_Properties


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


def First_Modal_Analysis(data: dict):
    """
    Larger analysis tool to determine the first modes in longitudinal, tangential, and radial orientation
    Uses the H&H formulation on page 129
    """
    frmt                = "{:<50} {:<10.3f} {:<10} {:<}"
    frmt2               = "{:<50} {:<10} {:<10} {:<}"

    x = data["E"]["x"]
    if x is None:
        x, y, a = build_nozzle(data=data)
        # cooling_geometry(dic=data)

    else:
        y = data["E"]["y"]
        a = data["E"]["a"]

    eps                 = data["E"]["aspect_ratio"]
    ind                 = np.where(eps == 1.0)[0][0]
    eps[:ind]           *= -1
    isentropic_nozzle_flow(eps=eps, data=data)

    T_c_static = data["Flow"]["T"][0]
    d_c = data["E"]["y"][0] * 2
    gamma_type = type(data["H"]["gamma"])
    R_c = data["H"]["R"]
    L_c = data["E"]["Lc"]

    if gamma_type == np.ndarray:
        gamma_c = data["H"]["gamma"][0]
    else:
        gamma_c = data["H"]["gamma"]


    a_chamber = np.sqrt(gamma_c * T_c_static * R_c)


    f_long = a_chamber / (2 * L_c)
    f_tang = 0.59 * a_chamber / (d_c)
    f_radi = 1.22 * a_chamber / d_c

    print("=" * 72, f"{'|':<}")
    print(f"{'First Modal Frequencies':^70} {'|':>3}")
    print(frmt.format("Longitudinal", f_long, "Hz", "|"))
    print(frmt.format("Tangential", f_tang, "Hz", "|"))
    print(frmt.format("Radial", f_radi, "Hz", "|"))

    # ======================= #
    # == INJECTOR ANALYSIS == #
    # ======================= #

    # Goal is to minimize the mass flow rate change and therefor chamber pressure change
    # Pressure drop to minimize dm/m
    # This does not use modal or frequency based analysis
    P_c_stag = data["E"]["Pc"]
    dP_injector = data["Injector"]["dP"]
    mdot = data["E"]["mdot"]

    # Injector delay phase check
    tau_eff = data["Injector"]["tau_eff"]


    print("=" * 72, f"{'|':<}")
    print(f"{'Injector Delay Phase Check':^70} {'|':>3}")
    if tau_eff is not None:
        phi_long = 2 * np.pi * f_long * tau_eff
        phi_tang = 2 * np.pi * f_tang * tau_eff
        phi_radi = 2 * np.pi * f_radi * tau_eff
        phis = {"Longitudinal": phi_long, "Tangential": phi_tang, "Radial": phi_radi}
        print(frmt2.format("A value > 1 requires a design change", "", "", "|"))
        for k,v in phis.items():
            print(frmt.format(f"     {k}", v, "", "|"))

    else:
        phi_long, phi_tang, phi_radi = 0, 0, 0
        print(frmt2.format("     [ERROR] Effective TAU not computed", "-", "-", "|"))

    # Feed system acoustic resonance
    # Determines if the feed wave is close in resonant frequency to the longitudinal mode
    # Speed of fluid assumed to be speed of sound of the fluid
    L_line_fuel = data["Injector"]["L_fuel"]
    L_line_ox = data["Injector"]["L_ox"]

    print("=" * 72, f"{'|':<}")
    print(f"{'Feed System Acoustic Resonance':^70} {'|':>3}")

    if L_line_ox is not None and L_line_fuel is not None:
        gamma_fuel = data["F"]["gamma"]
        temp_fuel = data["F"]["T"]
        R_fuel = data["F"]["R"]

        gamma_ox = data["O"]["gamma"]
        temp_ox = data["O"]["T"]
        R_ox = data["O"]["R"]

        c_feed_fuel = np.sqrt(gamma_fuel * temp_fuel * R_fuel)
        c_feed_ox = np.sqrt(gamma_ox * temp_ox * R_ox)

        f_feed_fuel = c_feed_fuel / (4 * L_line_fuel)
        f_feed_ox = c_feed_ox / (4 * L_line_ox)

        fuel_percent = np.abs(f_feed_fuel - f_long) / f_long
        if fuel_percent < 0.15:
            print(frmt.format("Fuel Feed back is similar resonance!", fuel_percent*100, "%", "|"))
            print(frmt.format("     Fuel", f_feed_fuel, "Hz", "|"))
            print(frmt.format("     Longitudinal", f_long, "Hz", "|"))
            print(frmt2.format("     Recommended to change oxidizer line length!", "", "", "|"))
        else:
            print(frmt2.format("Fuel feed back is different enough", "", "", "|"))
            print(frmt.format("     Fuel", f_feed_fuel, "Hz", "|"))
            print(frmt.format("     Longitudinal", f_long, "Hz", "|"))

        ox_percent = np.abs(f_feed_ox - f_long) / f_long
        if ox_percent < 0.15:
            print(frmt.format("Fuel Feed back is similar resonance!", ox_percent*100, "%", "|"))
            print(frmt.format("     Ox", f_feed_ox, "Hz", "|"))
            print(frmt.format("     Longitudinal", f_long, "Hz", "|"))
            print(frmt2.format("     Recommended to change oxidizer line length!", "", "", "|"))
        else:
            print(frmt2.format("Oxidizer feed back is different enough", "", "", "|"))
            print(frmt.format("     Ox", f_feed_ox, "Hz", "|"))
            print(frmt.format("     Longitudinal", f_long, "Hz", "|"))

    else:
        print(frmt2.format("     [ERROR] No line lengths present!", "-", "-", "|"))


    # Helmholtz resonator, used to determine the cavity size to dampen specific frequencies
    A_V_L_long = (f_long * 2 * np.pi / a_chamber)**2
    A_V_L_tang = (f_tang * 2 * np.pi / a_chamber)**2
    A_V_L_radi = (f_radi * 2 * np.pi / a_chamber)**2

    print("=" * 72, f"{'|':<}")
    print(f"{'Helmholtz Resonator Chamber Dimension Constant':^70} {'|':>3}")
    print(frmt2.format("Given is AV/L for the resonator", "", "", "|"))
    print(frmt2.format("Not necessary unless above results are un-fixable", "", "", "|"))
    print(frmt.format("     Longitudinal", A_V_L_long, "", "|"))
    print(frmt.format("     Tangential", A_V_L_tang, "", "|"))
    print(frmt.format("     Radial", A_V_L_radi, "", "|"))




    return f_long, f_tang, f_radi


def Startup_Analysis(data: dict):

    x = data["E"]["x"]
    if x is None:
        x, y, a = build_nozzle(data=data)
        # cooling_geometry(dic=data)

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


def CoolantSizingGuide(data: dict, fos_temp: float = 0.95, deposit_hg: float = 0, coolant_bulk_temp: float = 350,
                       tol: float = 1e-6, display:bool = True):
    """
    Drives the requirements for a properly cooled engine relying ONLY on regenerative cooling
    Derived from chapter 4 of H&H and specifically problem 4
    :param data: standard dictionary of info
    :param fos_temp: percentage of wall max temperature (0-1.0)
    :param deposit_hg: heat transfer coefficient of the coking (typically 3.82e-4 for throat), remove for ideal
    :param coolant_bulk_temp: coolant bulk temperature at throat
    """
    frmt                = "{:<50} {:<10.3f} {:<10} {:<}"
    frmt2               = "{:<50} {:<10} {:<10} {:<}"
    frmte               = "{:<50} {:<10.3e} {:<10} {:<}"

    x = data["E"]["x"]
    if x is None:
        x, y, a = build_nozzle(data=data)
    Fluid_Properties(dic=data, coolant_only=True)
    HotGas_Properties(dic=data, channel=True)

    max_wall_temp = data["W"]["solidus"] * fos_temp
    geom = data["C"]["Type"].lower()


    if display:
        print("=" * 72, f"{'|':<}")
        print(f"{'Generated Ideal Cooling Geometry Based On H&H':^70} {'|':>3}")
        print("- " * 36, f"{'|':<}")
        print(frmt.format("Max Wall Temp Used", max_wall_temp, "K", "|"))
        print(frmt2.format("Geometry Type", geom.title(), "", "|"))

    mdot = data["F"]["mdot"]
    Pc = data["E"]["Pc"]
    gamma = data["H"]["gamma"]
    R = data["H"]["R"]
    Tc = data["E"]["Tc"]
    mu = data["H"]["mu"][1]
    cstar_act = data["H"]["cstar"]
    Dt = data["E"]["r_throat"]*2
    i = data["C"]["pass"]
    t_wall = data["W"]["thickness"]

    # Hot gas information
    cstar_idl = np.sqrt(gamma*Tc*R) / (gamma*(2/(gamma+1))**((gamma+1)/(2*(gamma-1))))
    correction_factor = cstar_act / cstar_idl
    Tc_adjusted = Tc * correction_factor**2
    cp = gamma*R / (gamma-1)
    Pr = 4*gamma / (9*gamma - 5)
    mean_throat_radius = (data["E"]["r_exit"] + data["E"]["r_entry"]) / 2

    # Gas side heat flux using Bartz correlation
    # Only concerned with throat as this is the highest heat flux
    # Therefor all corrections and area corrections become 1.0
    h_hg = (0.026/Dt**0.2) * (mu**0.2*cp/Pr**0.6) * (Pc/cstar_act)**0.8 * (Dt/mean_throat_radius)**0.1
    h_hg = 1 / (1/h_hg + deposit_hg)


    # Required heat flux
    q_req = (Tc_adjusted - max_wall_temp) * h_hg

    # Wall properties
    k_wall = data["W"]["k"]
    t_wall_coolant = data["C"]["wall_thickness"]
    if geom == "circle":
        T_wall_coolant = max_wall_temp - (q_req * t_wall_coolant / k_wall)
    elif geom == "square":
        T_wall_coolant = max_wall_temp - (q_req * t_wall / k_wall)

# Coolant midpoint bulk temp
    # Coolant temp around throat region
    h_coolant = q_req / (T_wall_coolant - coolant_bulk_temp)
    cool_orig = data["F"]["T"]
    data["F"]["T"] = T_wall_coolant
    Fluid_Properties(dic=data, coolant_only=True)
    mu_wall_coolant = data["F"]["mu"]

    data["F"]["T"] = coolant_bulk_temp
    Fluid_Properties(dic=data, coolant_only=True)
    mu_bulk_coolant = data["F"]["mu"]
    cp_bulk_coolant = data["F"]["cp"]
    k_bulk_coolant = data["F"]["k"]
    data["F"]["T"] = cool_orig

    # # Sample Question verification
    # mdot = 827
    # h_coolant = 0.0075
    # mu_bulk_coolant = 4.16e-5
    # k_bulk_coolant = 1.78e-6
    # cp_bulk_coolant = 0.5
    # mu_wall_coolant = 0.419e-5
    # Dt = 24.9
    # t_wall_coolant = 0.02
    # geom = "circle"
    # i = 1

    term1_no_d = (mdot*i/mu_bulk_coolant)**0.8
    term2 = (mu_bulk_coolant * cp_bulk_coolant / k_bulk_coolant)**0.4
    term3 = (mu_bulk_coolant/mu_wall_coolant)**0.14

    nu_right = 0.0214*term1_no_d*term2*term3
    nu_left = h_coolant / k_bulk_coolant

    def secant_d_residual(d_i, geom_i):
        c_i = None
        if geom_i == "circle":
            N_i = np.pi * (Dt + 0.8*(d_i + 2*t_wall_coolant)) / (d_i + 2*t_wall_coolant)
            A_i = np.pi * d_i**2 / 4
            Dh = d_i
            c_i = Dh / N_i / A_i
        elif geom_i == "square":
            # N_i = np.pi * (Dt + 0.8*(d_i + 2*t_wall_coolant)) / (d_i + 2*t_wall_coolant)
            # N_i = np.pi * Dt / (d_i + t_wall_coolant)
            N_i = np.pi * (Dt + 2 * t_wall + d_i) / (2 * d_i)
            A_i = d_i ** 2
            Dh = d_i
            c_i = Dh / N_i / A_i

        left_side = nu_left * d_i
        right_side = nu_right * c_i**0.8
        return left_side - right_side

    d_0 = 0.0005
    # Small secant offset
    d = d_0 * 1.1
    # Iterate to find the true value of the diameter
    f_0 = secant_d_residual(d_0, geom_i=geom)

    for i in range(100):
        f = secant_d_residual(d, geom_i=geom)

        if abs(f) < tol:
            break

        # Secant update
        d_1 = d - f * (d-d_0)/(f-f_0 + 1e-12)

        # Physical guardrail
        d_1 = max(d_1, 1e-6)
        # Update
        d_0, f_0 = d, f
        d = d_1

    d_inner = round(d,6)


    if geom == "circle":
        N = np.pi * (Dt + 0.8 * (d_inner + 2 * t_wall_coolant)) / (d_inner + 2 * t_wall_coolant)
        N = np.floor(N)
        d_outer = Dt / (N / np.pi - 0.8)
        d_inner = d_outer - 2 * t_wall_coolant
        data["C"]["spacing"] = d_inner
        data["C"]["height"] = d_inner
        data["C"]["num_ch"] = N
        data["C"]["throat_bulk_temp"] = coolant_bulk_temp
        if display:
            print(frmt.format("Number of Channels", N, "", "|"))
            print(frmt.format("Inner Diameter", d_inner*1000, "mm", "|"))
            print(frmt.format("Outer Diameter", d_outer*1000, "mm", "|"))
            print(frmt.format("Wall Thickness", t_wall_coolant*1000, "mm", "|"))
            print(frmte.format("Channel Area", np.pi * d_inner**2/4, "m", "|"))
            print(frmt.format("Mass Flow Per Channel", data["F"]["mdot"] / data["C"]["num_ch"] * 1000, "g/s", "|"))
            print(frmt.format("Coolant Bulk Temp", coolant_bulk_temp, "K", "|"))

    elif geom == "square":
        N = np.pi * (Dt + 2*t_wall + d_inner) / (2*d_inner)
        N = np.floor(N)
        d = np.pi * (Dt + 2*t_wall) / (2*N - np.pi)
        data["C"]["spacing"] = d
        data["C"]["height"] = d
        data["C"]["depth"] = d
        data["C"]["num_ch"] = N
        data["C"]["throat_bulk_temp"] = coolant_bulk_temp
        if display:
            print(frmt.format("Number of Channels", N, "", "|"))
            print(frmt.format("Edge Length", d*1000, "mm", "|"))
            print(frmt.format("Edge Depth", d*1000, "mm", "|"))
            print(frmt.format("Fin Thickness", d*1000, "mm", "|"))
            print(frmt.format("Wall Thickness", t_wall*1000, "mm", "|"))
            print(frmte.format("Channel Area", d_inner**2, "m", "|"))
            print(frmt.format("Mass Flow Per Channel", data["F"]["mdot"] / data["C"]["num_ch"] * 1000, "g/s", "|"))
            print(frmt.format("Coolant Bulk Temp", coolant_bulk_temp, "K", "|"))

def non_throat_cooling_geom(data: dict):
    # The goal of this is to keep the area and fin thickness constant at each step
    # These values are going to stay constant
    N = data["C"]["num_ch"]
    t_f = data["C"]["spacing"]
    A = data["C"]["height"] * data["C"]["depth"]
    D = data["E"]["y"] * 2

    w = (np.pi * D / N) - t_f
    d = A / w

    data["C"]["depth_arr"] = d
    data["C"]["width_arr"] = w

def film_cooling(data: dict):
    pass





if __name__ == "__main__":

    CoolantSizingGuide(data={})



