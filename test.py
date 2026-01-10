import numpy as np

def heat_transfer_solver(data: dict):

    # ============ #
    # == SOLVER == #
    # ============ #
    energy_method = data["Solver"]["EnergyMethod"]

    """Engine Geometry"""
    x = data["E"]["x"]
    y = data["E"]["y"]
    N = len(x)

    """Channel Geometry"""
    depth = data["C"]["depth_arr"]
    width = data["C"]["width_arr"]

    """dx Setup"""
    dx_seg                      = np.diff(x)
    dx_i_arr                    = np.empty(N, dtype=float)
    dx_i_arr[:-1]               = dx_seg
    dx_i_arr[-1]                = dx_seg[-1]

    # By this point the stagnation conditions are already established
    # The Coolant channel and pressure drops have been calculated
    # The following arrays should NOT be changing within this function

    """Hot Gas Properties"""
    Tc          = data["E"]["Tc"]
    M_hg        = data["Flow"]["M"]
    gamma_hg    = data["H"]["gamma"]
    mu_hg       = data["H"]["mu"]
    k_hg        = data["H"]["k"]
    cp_hg       = data["H"]["cp"]

    """Coolant Properties"""
    mdot = data["F"]["mdot"]



    # These are initial conditions for the first slice ONLY
    # All other slices will be set at the end of the previous slice
    T_c = data["F"]["T"]
    P_c = data["F"]["P"]
    rho_c = data["F"]["rho"]
    h_c = data["F"]["H"]




    # Iterate through each slice
    for i in range(N-1, -1, -1):

        progress = (N - 1 - i) / (N - 1) * 100
        print(f"\rCooling solver progress: {progress:5.1f}%", end="", flush=True)

        # Slice radius
        y_i = y[i]

        # Slice thickness
        dx_i = dx_i_arr[i]
