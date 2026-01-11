import numpy as np
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import PropsSI


def heat_transfer_solver(data: dict, max_iter=50):

    # ============ #
    # == SOLVER == #
    # ============ #
    energy_method = data["Solver"]["EnergyMethod"]

    # == Engine Geometry == #
    x = data["E"]["x"]
    y = data["E"]["y"]
    U = data["E"]["U"]
    N = len(x)

    # == Channel Geometry == #
    depth:  dict        = data["C"]["depth_arr"]
    width:  dict        = data["C"]["width_arr"]
    num_ch: float       = data["C"]["num_ch"]

    # == dx Setup == #
    dx_seg                      = np.diff(x)
    dx_i_arr                    = np.empty(N, dtype=float)
    dx_i_arr[:-1]               = dx_seg
    dx_i_arr[-1]                = dx_seg[-1]

    """By this point the stagnation conditions are already established
    The Coolant channel and pressure drops have been calculated
    The following arrays should NOT be changing within this function"""

    # == Hot Gas Properties == #
    Tc:         float   = data["E"]["Tc"]
    P_hg:       dict    = data["Flow"]["P"]
    T_hg:       dict    = data["Flow"]["T"]
    M_hg:       dict    = data["Flow"]["M"]
    gamma_hg:   dict    = data["H"]["gamma"]
    mu_hg:      dict    = data["H"]["mu"]
    k_hg:       dict    = data["H"]["k"]
    cp_hg:      dict    = data["H"]["cp"]
    H_hg:       dict    = data["Flow"]["H"]
    mdot_hg:    float   = data["E"]["mdot"]

    # == Coolant Properties == #
    coolant:    str     = data["F"]["Type"]
    mdot:       float   = data["F"]["mdot"]

    """These are initial conditions for the first slice ONLY
    All other slices will be set at the end of the previous slice
    These are the stagnation conditions however"""

    # == Hot Gas Properties == #
    H_hg_0: float = data["H"]["H0"]
    C_hg: float   = 0.023

    # == Coolant Properties == #
    T_c = data["F"]["T"]
    P_c = data["F"]["P"]
    rho_c = data["F"]["rho"]
    H_c_0 = data["F"]["H"]

    # == Wall Properties == #
    T_wall_hg = data["Wall"]["InitialTemp"]

    """These are all the storage items to keep track of data"""
    # ==  == #
    H_c_0_arr:  np.ndarray  = np.zeros(N, dtype=float)



    # Iterate through each slice
    for i in range(N-1, -1, -1):

        progress = (N - 1 - i) / (N - 1) * 100
        print(f"\rCooling solver progress: {progress:5.1f}%", end="", flush=True)

        # Slice radius
        y_i = y[i]

        # Slice Position
        x_i = x[i]

        # Slice thickness
        # dx_i = dx_i_arr[i]
        dx_i = np.sqrt((y[i] - y[i-1])**2 + (x[i] - x[i-1])**2)

        if i != N-1:
            H_c_0 = H_c_0_arr[i]

        # Iterate multiple times to get the proper values
        for j in range(max_iter):
            # == HOT GAS == #
            H_wall_hg = H_hg[i] + cp_hg[i]*(T_wall_hg - T_hg[i])

            # Hot gas reference enthalpy
            H_hg_Ref = 0.5 * (H_wall_hg + H_hg[i]) + 0.18 * (H_hg_0 - H_hg[i])  # Place in loop

            # Hot gas reference temperature
            T_hg_Ref = CP.PropsSI("T", "P", P_hg, "H", H_hg_Ref)

            # Hot gas Prandtl reference number
            Pr_hg_Ref = cp_hg[i]*mu_hg[i]/k_hg[i]

            # Hot gas adiabatic wall temp
            H_hg_aw = H_hg[i] + (Pr_hg_Ref**(1/3) * (H_hg_0 - H_hg[i]))

            # Hot gas reynolds number reference
            Re_hg_Ref = 4 * mdot_hg * T_hg[i] / (np.pi * 2*y_i * mu_hg[i] * T_hg_Ref)

            # Hot gas side heat transfer coefficient
            h_hg = (C_hg * k_hg[i] / (y_i*2)) * Re_hg_Ref**0.8 * Pr_hg_Ref**0.3

            # == COOLANT == #

            # Coolant velocity
            v_c = mdot / (rho_c * depth[i] * width[i] * num_ch)

            # Static enthalpy
            H_c = H_c_0 - (v_c**2 / 2)

            # Some other stuff




            # Update stagnation enthalpy
            # H_c_0 = H_c_0_arr[i+1] + ((q + q_arr[i+1) * dx_i / (2 * mdot))




