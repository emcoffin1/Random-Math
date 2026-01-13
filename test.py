import numpy as np
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import PropsSI, AbstractState


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
    rho_hg:     dict    = data["Flow"]["rho"]

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
    T_c_0:      float       = data["F"]["T"]
    P_c_0:      float       = data["F"]["P"]
    rho_c_0:    float       = data["F"]["rho"]
    H_c_0:      float       = data["F"]["H"]
    C_c:        float       = 1
    g_c:        float       = 1


    # == Wall Properties == #
    T_wall_hg:  float       = data["Wall"]["InitialTemp"]
    T_wall_c:   float       = data["Wall"]["InitialTemp"]

    """These are all the storage items to keep track of data"""
    # == Stagnation Storage == #
    H_c_0_arr:  np.ndarray  = np.zeros(N, dtype=float)

    # == Static Storage == #
    P_c_arr:    np.ndarray  = np.zeros(N, dtype=float)
    H_c_arr:    np.ndarray  = np.zeros(N, dtype=float)
    rho_c_arr:  np.ndarray  = np.zeros(N, dtype=float)
    Re_c_arr:   np.ndarray  = np.zeros(N, dtype=float)
    Re_c_Ref_arr:   np.ndarray  = np.zeros(N, dtype=float)
    v_c_arr:    np.ndarray  = np.zeros(N, dtype=float)

    # == Energy Storage == #
    q_c_arr:    np.ndarray  = np.zeros(N, dtype=float)

    # == Individual Values == #
    rho_c:      float       = np.nan


    st = AbstractState("HEOS", coolant)

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
        dx_i = np.sqrt((y[i] - y[i+1])**2 + (x[i] - x[i+1])**2)

        # If first slice, static is equal to stagnation
        # Otherwise this will be set after convergence



        # Iterate multiple times to get the proper values
        for j in range(max_iter):

            # If not the first slice, and then if first iteration or not
            if i != N-1:
                if j == 1:
                    rho_c: float = float(rho_c_arr[i+1])
                else:
                    # This is derived directly from the previous iteration, no manual adjusting of any kind
                    rho_c: float = CP.PropsSI("rhomass", "P", P_c_arr[i+1], "H", H_c_arr[i+1])

            else:
                if j == 1:
                    H_c = H_c_0
                    P_c = P_c_0
                    rho_c = rho_c_0

            # If first iteration, update stagnation enthalpy with eqn 9
            if j == 1:
                H_c_0 = H_c_0_arr[i+1] + (q_c_arr[j+1] * dx_i / mdot)

            else:
                H_c_0 = H_c_0_arr[i+1] + ((q_c_arr[j] + q_c_arr[j+1]) * dx_i / mdot)

            # Coolant velocity
            v_c = mdot / (rho_c * depth[i] * width[i] * num_ch)

            # Static enthalpy
            H_c = H_c_0 - (v_c**2 / 2)

            # Coolant Reynolds values (static and reference)
            mu_c: float = CP.PropsSI("viscosity", "P", P_c_arr[i + 1], "H", H_c_arr[i + 1])
            Re_c: float = mdot * y_i / (depth[i] * width[i] * num_ch * mu_c)

            Re_c_Ref: float = Re_c * (rho_hg[i] / rho_c) * (mu_c / mu_hg[i])

            # Average coolant reynolds number for better accuracy
            # Do not use if first slice
            if j != N-1:
                Re_c_avg = 0.5 * (Re_c_arr[i] + Re_c_arr[i+1])
                Re_c_Ref_avg = 0.5 * (Re_c_Ref[i] + Re_c_Ref_arr[i+1])
            else:
                Re_c_avg = Re_c
                Re_c_Ref_avg = Re_c_Ref

            # Friction factor using Moody diagram
            C1 = C_c / 0.023
            if Re_c_Ref_avg < 2.2e3:
                f_c = 64 / Re_c_Ref_avg
            elif 2.2e3 <= Re_c_Ref_avg < 10e4:
                f_c = 4*C1*(0.0014+ (0.125/Re_c_Ref_avg**0.32))
            else:
                f_c = 0.778*C_c*Re_c_Ref_avg**-0.1021

            # Viscous Pressure drop (darcy)
            dP_c_f = f_c/(4*g_c) * ((rho_c + rho_c_arr[i+1])/(y_i + y[i+1])) * (v_c**2 + v_c_arr[i+1]**2) * dx_i

            # Momentum pressure drop
            dP_c_M = ((2 / ((N*depth[i+1]*width[i+1]) + (N*depth[i]*depth[i+1]))) *
                      mdot**2/g_c *
                      ((1 / (rho_c*depth[i]*width[i]*N)) - (1 / (rho_c_arr[i+1]*depth[i+1]*width[i+1]*N))))

            # New coolant pressure (eqn 15)
            P_c = P_c_arr[i+1] - (dP_c_f + dP_c_M)

            # Updated coolant wall properties
            st.update(CP.PT_INPUTS, P_c, T_wall_c)
            cp_w_c = st.cpmass()
            mu_w_c = st.viscosity()
            k_w_c = st.conductivity()
            P_c_w = st.p()
            H_c_w = st.hmass()

            # Coolant reference enthalpy (eqn 16)
            H_c_Ref = 0.5*(H_c + H_c_w) + 0.194*(H_c_0 - H_c)

            # Update coolant reference properties
            st.update(CP.PSmass_INPUTS, P_c, H_c_Ref)



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


            # Some other stuff




            # Update stagnation enthalpy
            # H_c_0 = H_c_0_arr[i+1] + ((q + q_arr[i+1) * dx_i / (2 * mdot))


        # Next stations static point is the previous stations converged stagnation??



