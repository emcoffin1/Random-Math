import numpy as np
from NozzleDesign import build_nozzle
from GasProperties import *

def friction_factor(Re: float) -> float:
    """
    Darcy friction factor for smooth ducts
    https://www.idrologia.unimore.it/pizzileo/web-archive/af/lezione1/Darcy%20friction%20factor%20formulae.pdf
    """
    if Re <= 0 or not np.isfinite(Re):
        return np.nan
    if Re < 2300:
        # Laminar flow friction factor
        return 64 / Re
    if Re < 10e5:
        # Blasius Correlations without correction
        return 0.3146 * Re ** (-0.25)
    else:
        raise ValueError("Reynolds number is too high in cooling channels for current model: ", Re)


def dp_total_for_width(info: dict, width:float, alpha=0.8, beta=1.05) -> float:

    # Coolant properties
    mdot = info["F"]["mdot"]
    rho = info["F"]["rho"]
    mu = info["F"]["mu"]

    # Channel / Wall properties
    N_ch = info["C"]["num_ch"]
    height = info["C"]["height"]
    spacing = info["C"]["spacing"]
    thickness_w = info["W"]["thickness"]

    x = np.asarray(info["E"]["x"], dtype=float)
    y = np.asarray(info["E"]["y"], dtype=float)

    # Ensure we actually have values to use
    if width <= 0:
        return np.inf


    # Channel area, with rounding factor
    A_ch = np.pi / N_ch * (height**2 + 2*y*height + 2*thickness_w*height) - (spacing*height)
    # Wetted perimeter
    # P_w = beta * (2.0 * height + width)
    P_w = 2 * ((np.pi / N_ch * (2 * y + 2 * thickness_w + height)) - spacing + height)
    # Hydraulic diameter
    Dh = 4.0 * A_ch / P_w

    if A_ch <= 0 or Dh <= 0:
        return np.inf

    # Flow velocity in each channel from rho v a
    v = mdot / (rho * N_ch * A_ch)

    dx = np.diff(x)
    if np.any(dx <= 0):
        raise ValueError("x must be increasing for dp integration")

    Re = rho * v * Dh / mu
    f = friction_factor(Re)

    if not np.isfinite(f):
        return np.nan

    dp = np.sum(f * (dx / Dh) * (rho * v**2 / 2))
    # dp = 0
    # for dx_i in dx:
    #     Re = rho * v * Dh / mu
    #     f = friction_factor(Re)
    #     dp += f * (dx_i / Dh) * (rho * v**2 / 2.0)

    info["C"]["A_ch"] = A_ch
    info["C"]["P_w"] = P_w
    info["C"]["Dh"] = Dh
    info["C"]["v"] = v
    info["C"]["Re"] = Re

    return dp

def solve_width_for_dp(info: dict, dp_target: float, alpha=0.8, beta=1.05, w_min=1e-8, w_max=5e-2, tol=1e-3,
                       max_iter=80, max_attempts=5):
    """
    Uses bisection method to determine channel geometry to achieve a specific pressure dorp through the regen channels
    :param info:
    :param dp_target:
    :param alpha:
    :param beta:
    :param w_min:
    :param w_max:
    :param tol:
    :param max_iter:
    :return:
    """
    # Store in case
    orig = info["C"]["num_ch"]

    # Initial values
    lo, hi = w_min, w_max
    dp_lo = dp_total_for_width(info=info, width=lo, alpha=alpha, beta=beta)
    dp_hi = dp_total_for_width(info=info, width=hi, alpha=alpha, beta=beta)

    # Ensure bracket: dp(lo) should be > target, dp(hi) < target typically
    if not (np.isfinite(dp_lo) and np.isfinite(dp_hi)):
        raise ValueError("Non-finite dp encountered; check properties/geometry.")

    if dp_lo < dp_target:
        raise ValueError(
            f"Target dp too high: dp({lo * 1e3:.3f} mm)={dp_lo / 1e5:.2f} bar"
        )
    if dp_hi > dp_target:
        raise ValueError(
            f"Target dp too low: dp({hi * 1e3:.3f} mm)={dp_hi / 1e5:.2f} bar"
        )


    for _ in range(max_iter):
        mid = 0.5*(hi + lo)
        dp_mid = dp_total_for_width(info=info, width=mid, alpha=alpha, beta=beta)

        # Find percent error from target
        if abs(dp_mid - dp_target) / dp_target < tol:
            return mid, dp_mid

        # Adjust brackets
        if dp_mid > dp_target:
            lo = mid
        else:
            hi = mid

    # The attempts failed, recompute and send the original
    mid = 0.5 * (hi + lo)
    return mid, dp_total_for_width(info=info, width=mid, alpha=alpha, beta=beta)


def thickness_bi(info: dict, guess: float):
    """Solves for t_c using a guess"""
    mdot = info["F"]["mdot"]
    rho = info["F"]["rho"]
    mu = info["F"]["mu"]

    fin_height = info["C"]["height"]
    throat_radius = info["E"]["r_throat"]
    thickness_wall = info["W"]["thickness"]

    P_wet = 2 * (guess/(throat_radius+thickness_wall)*(2*throat_radius + 2*thickness_wall + fin_height) - guess + fin_height)
    A = guess * ((fin_height**2 + 2*throat_radius*fin_height + 2*thickness_wall*fin_height) / (throat_radius+thickness_wall) - fin_height)
    Re = 4200 # almost guaranteed change
    v = Re * mu * P_wet / (4 * rho * A)

    term1 = mdot / (rho * v * fin_height)
    term2 = np.pi * (fin_height + 2*throat_radius + 2*thickness_wall)

    return term2 - term1



def solve_fin_width(info: dict, lo=1e-6, hi=5e-2, max_iteration=80, tol=1e-8):
    """Uses bisection method to solve for t_c"""
    f_lo = thickness_bi(info=info, guess=lo)
    f_hi = thickness_bi(info=info, guess=hi)

    for _ in range(max_iteration):

        print(lo, f_lo, hi, f_hi)
        mid = 0.5*(lo + hi)
        f_mid = thickness_bi(info=info, guess=mid)


        if abs(f_mid) < tol or (hi - lo) < tol:
            return mid

        if f_lo * f_mid < 0:
            hi = mid
            f_hi = f_mid

        else:
            lo = mid
            f_lo = f_mid

    return round(0.5 * (lo + hi), 6)

def solve_fin_width2(info:dict, guess=1e-3, max_iteration=80, tol=1e-8):

    for _ in range(max_iteration):
        pass




if __name__ == "__main__":
    info = {"CEA": True,
            "plots": "no",
            "dimensions": 1,    # Complexity of heat transfer
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
            },
            "W": {
                # "Type": "SS 316L",
                # "Type": "Tungsten",
                "Type": "Copper Chromium",
                "thickness": 0.02
            },
            "C": {
                "Type": "Square",
                "thickness": 0.02,  # Wall thickness
                "width": None,     # Diameter if circle geometry
                "spacing": 0.02,   # Fin thickness -- space between channels
                "height": 0.02,     # Channel height
                "num_ch": 20,
                "dP": 551581,
                "h": None,
                "Nu": None,
                "Re": None,
            },
            "Flow": {
                "x": None,
                "y": None,
                "a": None,
                "eps": None
            },


            }

    if info["CEA"]:
        # Run rocketcea
        HotGas_Properties(Pc=info["E"]["Pc"], fuel=info["F"]["Type"], ox=info["O"]["Type"], OF=info["E"]["OF"], dic=info)
        Fluid_Properties(dic=info)
        Material_Properties(dic=info)

    x, y, a = build_nozzle(data=info)
    print(solve_fin_width(info=info))
    print(min(y))

    # for w in [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]:
    #     x, y, a = build_nozzle(data=info)
    #     dp = dp_total_for_width(info, w)
    #     print(f"w={w * 1e3:.2f} mm -> dp={dp / 1e5:.2f} bar")







