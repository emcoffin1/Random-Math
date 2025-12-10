import numpy as np
from MachSolver import area_ratio_from_M
from _extra_utils import plot_engine
from scipy.interpolate import interp1d, splrep, splev, PchipInterpolator
import matplotlib.pyplot as plt


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
    i = np.abs(high-low) * 2
    theta = np.linspace(low, high, num=round(i))
    theta = np.radians(theta)
    x = 1.5 * Rt * np.cos(theta)
    y = 1.5 * Rt * np.sin(theta) + 1.5 * Rt + Rt

    return x, y


# def exit_section(Rt, theta_n):
#     """
#
#     :param Rt:
#     :param theta_n:
#     :return:
#     """
#     low = -90
#     high = np.degrees(theta_n) - 90.0
#     i = np.abs(high-low) * 2
#     theta = np.linspace(low, high, num=round(i))
#     theta = np.radians(theta)
#     x = 0.382 * Rt * np.cos(theta)
#     y = 0.382 * Rt * np.sin(theta) + 0.382 * Rt + Rt
#
#     return x, y

def exit_section(Rt, Nx, Ny):
    """
    Correct throat exit section:
    - Uses the standard 0.382 Rt throat circle
    - Ends exactly at the parabolic start point N (Nx, Ny)
    - Avoids forcing a tangent match (handled by Q in Rao geometry)
    """

    R = 0.382 * Rt
    Cx = 0.0
    Cy = Rt + R     # same as your original

    # Solve for final circle angle φ such that (x,y) = (Nx, Ny)
    # Nx = Cx + R cos φ
    # Ny = Cy + R sin φ
    dx = Nx - Cx
    dy = Ny - Cy

    phi_end = np.arctan2(dy, dx)

    # Start angle always -90° for throat circle start
    phi_start = -np.pi / 2

    # Generate arc
    num = int(abs(phi_end - phi_start) * 80)
    phi = np.linspace(phi_start, phi_end, num)

    x = Cx + R * np.cos(phi)
    y = Cy + R * np.sin(phi)

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


def get_angles(Rt, ar:float, bell_percent=0.8):
    """
    Extrapolates the angle for exit and for initial conditions
    :param Rt:              Throat Radius [m]
    :param ar:              Aspect Ratio []
    :param bell_percent:    Percent of nozzle [%]
    :return:
    """
    bp = bell_percent
    aratio      = np.array([3.5,   4,     5,   10,     20,    30,    40,    50,    100])

    theta_n_60  = np.array([25.5,  26.5,  28,  32,     35,    36.3,  37.5,  38,    40.25])
    theta_n_70  = np.array([23,    24,    25,  28.5,   31.3,  32.6,  33.6,  34.5,  36.5])
    theta_n_80  = np.array([21,    21.6,  23,  26.3,   27.9,  30,    31,    32.2,  33.75])
    theta_n_90  = np.array([19,    20,    21,  24.25,  27,    28.5,  29.3,  30,    32.5])
    theta_n_100 = np.array([18.5,  19,    20,  22.5,   25.5,  27,    28,    29,    32])

    theta_e_60  = np.array([21.8,  21,    19,  16,     14.5,  14,    13.5,  13,    12])
    theta_e_70  = np.array([18,    17,    16,  13,     12,    11.2,  10.8,  10.5,  9.75])
    theta_e_80  = np.array([14.8,  14,    13,  10.5,   9,     8.25,  8,     7.75,  7])
    theta_e_90  = np.array([12,    11,    10,  8,      7,     6.5,   6.25,  6,     6])
    theta_e_100 = np.array([10,    9,     8,   6,      5.25,  4.9,   4.75,  4.5,   4.25])

    # nozzle length
    f1 = ((np.sqrt(ar) - 1) * Rt)/ np.tan(np.radians(15))

    if bp == 0.6:
        theta_n = theta_n_60; theta_e = theta_e_60
        L = 0.6 * f1

    elif bp == 0.7:
        theta_n = theta_n_70; theta_e = theta_e_70
        L = 0.7 * f1

    elif bp == 0.8:
        theta_n = theta_n_80; theta_e = theta_e_80
        L = 0.8 * f1

    elif bp == 0.9:
        theta_n = theta_n_90; theta_e = theta_e_90
        L = 0.9 * f1

    else:
        theta_n = theta_n_100; theta_e = theta_e_100
        L = 1.0 * f1

    # find the nearest value using linear interp (not very accurate)
    theta_n = np.interp(np.log10(ar), np.log10(aratio), theta_n)
    theta_e = np.interp(np.log10(ar), np.log10(aratio), theta_e)

    return L, np.radians(theta_n), np.radians(theta_e)


def point_selection(Rt, eps, theta_n, theta_e, bell_percent=0.8) -> list:
    """
    Computes the nozzle curve points E, Q, N
    :param Rt:              Throat Radius [m]
    :param eps:             Expansion Ratio []
    :param theta_n:         Entry angle [rad]
    :param theta_e:         Exit angle [rad]
    :param bell_percent:    Percent of nozzle [%]
    :return:                list [Nx, Ny, Ex, Ey, Qx, Qy]
    """
    theta = theta_n - np.radians(90)
    Nx = 0.382 * Rt * np.cos(theta)
    Ny = 0.382 * Rt * np.sin(theta) + 0.382 * Rt + Rt

    bp = bell_percent
    Ex = bp * (np.sqrt(eps) - 1) * Rt / np.tan(np.radians(15))
    Ey = np.sqrt(eps) * Rt

    m1 = np.tan(theta_n)
    m2 = np.tan(theta_e)
    c1 = Ny - m1*Nx
    c2 = Ey - m2*Ex
    Qx = (c2 - c1)/(m1 - m2)
    Qy = (m1*c2 - m2*c1)/(m1 - m2)

    points = [Nx, Ny, Ex, Ey, Qx, Qy]
    return points


def area_conversion(y):
    r = y
    a = np.pi * r**2
    return a


def build_nozzle(data:dict, eps=None):
    """
    Build the full nozzle, and gives an option to plot the data in 2d or 3d
    :param data:    Dictionary of engine data
    :param eps:     Area ratio
    :return:        x, y, a
    """
    # Break apart info dict
    Pe, Pc, size, Rt, gamma, plots = (data["Pe"], data["Pc"], data["size"], data["Rt"], data["gamma"], data["plots"])

    # Pressure ratio
    P_r = Pc / Pe
    # Exit mach
    Me = exit_mach_from_p(P_r)
    # Expansion ratio
    if eps is None:
        eps = area_ratio_from_M(Me, gamma=gamma)

    # Engine Structure/shape
    L, n, e = get_angles(Rt, eps, bell_percent=size)
    points = point_selection(Rt, eps, n, e, bell_percent=size)
    xq, yq = quadratic_curve(points)
    xen, yen = entry_section(Rt)
    Nx, Ny = points[0], points[1]
    # xe, ye = exit_section(Rt, n)
    xe, ye = exit_section(Rt, Nx, Ny)

    # Build out full line
    x = np.concatenate((xen[::-1], xe, xq))
    y = np.concatenate((yen[::-1], ye, yq))

    # Get area for each
    a = area_conversion(y)

    if plots == "no":
        return x, y, a

    elif plots == "2D" or plots == "3D":
        plot_engine(x, y, plots)
        return x, y, a
