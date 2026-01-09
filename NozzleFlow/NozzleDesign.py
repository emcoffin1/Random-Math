import numpy as np
from MachSolver import area_ratio_from_M
from _extra_utils import plot_engine, convert_to_func
from GasProperties import HotGas_Properties

"""https://rrs.org/2023/01/28/making-correct-parabolic-nozzles/"""
"""Compare with this at some point"""




def throat_radius(flow: dict):
    mdot, Pc, gamma, R, Tc = flow["E"]["mdot"], flow["E"]["Pc"], flow["H"]["gamma"], flow["H"]["R"], flow["E"]["Tc"]
    At = (mdot / Pc) * np.sqrt(R * Tc / gamma) * ((gamma + 1) / 2) ** ((gamma + 1) / (2 * (gamma - 1)))
    Rt = np.sqrt(At / np.pi)
    return Rt


def exit_mach_from_p(P_r, gamma=1.4):
    P = P_r
    gr = (gamma - 1) / gamma
    term1 = 2 / (gamma - 1)
    term2 = P**gr - 1
    return np.sqrt(term1 * term2)


def entry_section(Rt, high=-135, low=-90):
    """
    Forms the curve leading into the throat region
    :param Rt:      Throat radius [m]
    :param high:    Highest angle (left) [deg]
    :param low:     Lowest angle (right) [deg]
    :return:        x, y
    """
    i = np.abs(high - low) * 2
    theta = np.linspace(low, high, num=round(i))
    theta = np.radians(theta)
    x = 1.5 * Rt * np.cos(theta)
    y = 1.5 * Rt * np.sin(theta) + 1.5 * Rt + Rt
    return x, y


def exit_section(Rt, theta_n):
    """
    Throat exit circular arc
    """
    low = -90
    high = np.degrees(theta_n) - 90.0
    i = np.abs(high - low) * 2
    theta = np.linspace(low, high, num=round(i))
    theta = np.radians(theta)
    x = 0.382 * Rt * np.cos(theta)
    y = 0.382 * Rt * np.sin(theta) + 0.382 * Rt + Rt
    return x, y


def quadratic_curve(points):
    """
    Plots the quadratic curve of the nozzle portion
    :param points:  List of N Q and E points
    :return:
    """
    t = np.linspace(0, 1, 10)
    Nx, Ny, Ex, Ey, Qx, Qy = points[0], points[1], points[2], points[3], points[4], points[5]

    x = ((1 - t) ** 2) * Nx + 2 * (1 - t) * t * Qx + (t ** 2) * Ex
    y = ((1 - t) ** 2) * Ny + 2 * (1 - t) * t * Qy + (t ** 2) * Ey

    return x, y


def get_angles(Rt, ar: float, bell_percent=0.8):
    """
    Extrapolates the angle for exit and for initial conditions
    :param Rt:              Throat Radius [m]
    :param ar:              Aspect Ratio []
    :param bell_percent:    Percent of nozzle [%]
    :return:
    """
    bp = bell_percent
    aratio = np.array([3.5, 4, 5, 10, 20, 30, 40, 50, 100])

    theta_n_60 = np.array([25.5, 26.5, 28, 32, 35, 36.3, 37.5, 38, 40.25])
    theta_n_70 = np.array([23, 24, 25, 28.5, 31.3, 32.6, 33.6, 34.5, 36.5])
    theta_n_80 = np.array([21, 21.6, 23, 26.3, 27.9, 30, 31, 32.2, 33.75])
    theta_n_90 = np.array([19, 20, 21, 24.25, 27, 28.5, 29.3, 30, 32.5])
    theta_n_100 = np.array([18.5, 19, 20, 22.5, 25.5, 27, 28, 29, 32])

    theta_e_60 = np.array([21.8, 21, 19, 16, 14.5, 14, 13.5, 13, 12])
    theta_e_70 = np.array([18, 17, 16, 13, 12, 11.2, 10.8, 10.5, 9.75])
    theta_e_80 = np.array([14.8, 14, 13, 10.5, 9, 8.25, 8, 7.75, 7])
    theta_e_90 = np.array([12, 11, 10, 8, 7, 6.5, 6.25, 6, 6])
    theta_e_100 = np.array([10, 9, 8, 6, 5.25, 4.9, 4.75, 4.5, 4.25])

    # nozzle length
    f1 = ((np.sqrt(ar) - 1) * Rt) / np.tan(np.radians(15))

    if bp == 0.6:
        theta_n = theta_n_60
        theta_e = theta_e_60
        L = 0.6 * f1

    elif bp == 0.7:
        theta_n = theta_n_70
        theta_e = theta_e_70
        L = 0.7 * f1

    elif bp == 0.8:
        theta_n = theta_n_80
        theta_e = theta_e_80
        L = 0.8 * f1

    elif bp == 0.9:
        theta_n = theta_n_90
        theta_e = theta_e_90
        L = 0.9 * f1

    else:
        theta_n = theta_n_100
        theta_e = theta_e_100
        L = 1.0 * f1

    # find the nearest value using linear interp (not very accurate)
    theta_n = np.interp(np.log10(ar), np.log10(aratio), theta_n)
    theta_e = np.interp(np.log10(ar), np.log10(aratio), theta_e)

    return L, np.radians(theta_n), np.radians(theta_e)


def point_selection(Rt, eps, theta_n, theta_e, bell_percent=0.8) -> list:
    """
    Computes the nozzle curve points E, Q, N
    """
    theta = theta_n - np.radians(90)
    Nx = 0.382 * Rt * np.cos(theta)
    Ny = 0.382 * Rt * np.sin(theta) + 0.382 * Rt + Rt

    bp = bell_percent
    Ex = bp * (np.sqrt(eps) - 1) * Rt / np.tan(np.radians(15))
    Ey = np.sqrt(eps) * Rt

    m1 = np.tan(theta_n)
    m2 = np.tan(theta_e)
    c1 = Ny - m1 * Nx
    c2 = Ey - m2 * Ex

    Qx = (c2 - c1) / (m1 - m2)
    Qy = (m1 * c2 - m2 * c1) / (m1 - m2)

    points = [Nx, Ny, Ex, Ey, Qx, Qy]
    return points


def area_conversion(y):
    r = y
    a = np.pi * r**2
    return a


def chamber_contraction(x, y, info: dict, Lc=0.05):
    """
    Chamber geometry derived from page 73 and 74 of H&H
    """

    theta = np.radians(30)

    y_conv = np.min(y)
    Rt = y_conv
    A_t = np.pi * Rt**2
    CR = info["E"]["CR"]
    r_c = Rt * np.sqrt(CR)
    A_c = Rt * CR

    f = info["F"]["Type"]
    o = info["O"]["Type"]
    if (f == "RP-1" or f=="Kerosene") and o == "LOX":
        Lstar = 1.143   # 45 in

    elif (f == "CH4") and o == "LOX":
        Lstar = 2.0
    else:
        Lstar = 1.0
        print("WARNING : Current propellant choices not available, defaulting to L* = ", Lstar)

    info["E"]["Lstar"] = Lstar


    # CR is the ratio of chamber area to throat area
    # Chamber length is essentially L* Ac / At
    V_c = Lstar * A_t
    L_c = ( (V_c / A_t) - 1/3*np.sqrt(A_t/np.pi) / np.tan(theta) * (CR**(1/3) - 1)) / CR

    info["E"]["Lc"] = L_c
    info["E"]["Ac"] = A_c
    x0 = x[0]

    x_ch = np.linspace(x0 - L_c, x0-Lc, 50)
    y_ch = np.full_like(x_ch, r_c)

    x_con = np.linspace(x0-Lc, x0, 50)
    y_con = Rt + 0.5*(r_c - Rt)*(1 + np.cos(np.pi*(x_con - (x0-Lc))/Lc))

    return x_ch, y_ch, x_con, y_con


def build_nozzle(data: dict, chamber=True):
    """
    Build the full nozzle and optionally plot
    """
    # Pressure ratio
    P_r = data["E"]["Pc"] / data["E"]["Pe"]

    # Exit mach
    Me = exit_mach_from_p(P_r)

    # Expansion ratio
    data["E"]["eps"] = area_ratio_from_M(Me, gamma=data["H"]["gamma"])
    data["E"]["Rt"] = throat_radius(flow=data)

    Pe, Pc, T, size, mdot, Rt, gamma, R, plots, eps = (
        data["E"]["Pe"], data["E"]["Pc"], data["E"]["Tc"], data["E"]["size"], data["E"]["mdot"], data["E"]["Rt"], data["H"]["gamma"],
        data["H"]["R"], data["Display"]["EnginePlot"], data["E"]["eps"]
    )

    # Engine Structure/shape
    L, n, e = get_angles(Rt, eps, bell_percent=size)
    points = point_selection(Rt, eps, n, e, bell_percent=size)

    xq, yq = quadratic_curve(points)
    xe, ye = exit_section(Rt, n)
    xen, yen = entry_section(Rt)

    # if not chamber:
    #     xen, yen = entry_section(Rt)
    #     # Build out full line
    #     x = np.concatenate((xen[::-1], xe, xq))
    #     y = np.concatenate((yen[::-1], ye, yq))
    #
    # else:
    #
    #     x = np.concatenate((xe, xq))
    #     y = np.concatenate((ye, yq))
    #
    #     x_ch, y_ch, x_con, y_con = chamber_contraction(x=xe, y=ye, info=data)
    #
    #     x = np.concatenate((x_ch, x_con[:-1], x))
    #     y = np.concatenate((y_ch, y_con[:-1], y))

    if not chamber:
        x_ch = np.zeros(5)
        x_con = np.zeros(5)
        y_ch = np.zeros(5)
        y_con = np.zeros(5)

    else:
        x_ch, y_ch, x_con, y_con = chamber_contraction(x=xe, y=ye, info=data)

    # Full use of Rao Bell
    x = np.concatenate((x_ch, xen[:-1], xe, xq))
    y = np.concatenate((y_ch, yen[:-1], ye, yq))

    x, y = convert_to_func(x=x, y=y, save=False)

    a = area_conversion(y)
    r_throat = np.min(y)

    if round(r_throat,4) != round(Rt,4):
        raise ValueError(f"The throat radii do not match Rt: {Rt:.4f} -- r_throat:{r_throat:.4f}")

    a_t = min(a)
    data["E"]["aspect_ratio"] = a / a_t
    data["E"]["x"] = x
    data["E"]["y"] = y
    data["E"]["a"] = a
    data["E"]["r_throat"] = Rt
    data["E"]["r_exit"] = 0.382 * Rt
    data["E"]["r_entry"] = 1.5 * Rt

    N = len(x)
    dx_edge = np.diff(x)
    dx = np.zeros(N)

    dx[0] = dx_edge[0]
    dx[-1] = dx_edge[-1]
    dx[1:-1] = 0.5 * (dx_edge[1:] + dx_edge[:-1])
    data["E"]["dx"] = dx

    if np.any(data["E"]["aspect_ratio"] <= 0):
        raise ValueError("There is a negative value in the aspect ratio. FOUND IN DESIGN")


    # Recompute the values
    # HotGas_Properties(Pc=Pc, fuel=data["F"]["Type"], ox=data["O"]["Type"], OF=data["E"]["OF"], eps=data["E"]["eps"],dic=data)

    if plots == "no":
        # plt.plot(x, y)
        # plt.plot(x,-y)
        # plt.show()
        return x, y, a

    elif plots == "2D" or plots == "3D":
        plot_engine(y=y, x=x, type=plots)
        return x, y, a
