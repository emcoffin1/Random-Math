import numpy as np
import CoolProp.CoolProp as CP
from CoolProp.CoolProp import PropsSI, AbstractState
from NozzleFlow._extra_utils import Moody_plot
from NozzleFlow.GasProperties import Material_Properties


def heat_transfer_solver(data: dict, max_iter=50, tol=1e-4):
    def first_station_function(T_c_0, T_wall_c, T_wall_hg, i):
        """
        This function is designed to handle all iterations for the first slice
        This is done since the first iteration doesn't have any previous stations to reference
        To handle this, all conditions are considered to be the initial static condition
        This will also be handled by assuming that x=N is the initial static conditions,
        and we'll assume zero loss through this region
        """
        for j in range(max_iter):
            # == COOLANT == #
            # Geometric Properties
            Dh_c = 2 * depth[i] * width[i] / (depth[i] + width[i])

            # First pull the initial static conditions of the coolant
            st.update(CP.PT_INPUTS, P_c_0, T_c_0)
            cp_c_0 = st.cpmass()
            k_c_0 = st.conductivity()
            H_c_0 = st.hmass()
            rho_c_0 = st.rhomass()
            mu_c_0 = st.viscosity()

            # Determine the fluid behavior
            v_c_0 = mdot / (rho_c_0 * depth[i] * width[i] * num_ch)
            Re_c_0 = rho_c_0 * v_c_0 * Dh_c / mu_c_0
            Pr_c_0 = cp_c_0 * mu_c_0 / k_c_0

            # Determine heat transfer coefficient as piecewise of Pr
            if 0.1 <= Pr_c_0 <= 1.0:
                Nu_c_0: float = 0.02155 * Re_c_0 ** 0.8018 * Pr_c_0 ** 0.7095
            elif 1.0 < Pr_c_0 <= 3.0:
                Nu_c_0: float = 0.01253 * Re_c_0 ** 0.8413 * Pr_c_0 ** 0.6179
            else:
                Nu_c_0: float = 0.00881 * Re_c_0 ** 0.8991 * Pr_c_0 ** 0.3911

            h_c_httrans: float = Nu_c_0 * k_c_0 / Dh_c

            # === WALL === #
            T_avg = 0.5 * (T_wall_c + T_wall_hg)
            Material_Properties(dic=data, T=T_avg)

            h_w_httrans = k_wall / t_wall

            # == HOT GAS == #
            Pr_hg_0 = data["H"]["Pr"]
            recovery_hg = Pr_hg_0**(1/3)
            T_aw_i: float = T_hg[i] + (recovery_hg*(U[i]**2/ (2*cp_hg[i])))
            # T_aw_i = T_hg[i] + (recovery_hg*(Tc - T_hg[i]))
            Re_hg = 4 * mdot_hg / (np.pi * 2*y_i * mu_hg[i])

            h_hg_httrans = 0.026 * (k_hg[i]/(2*np.min(data["E"]["y"]))) * Re_hg**0.8 * Pr_hg_0**0.3

            # Compute all the heat fluxes to make sure they match
            H = (1/h_hg_httrans + 1/h_w_httrans + 1/h_c_httrans)**-1
            q_tot = H*(T_aw_i - T_c_0)

            # Hot gas side wall temp
            T_wg = T_aw_i - (q_tot/h_hg_httrans)

            # Coolant side wall temp
            T_wc = T_wg - (q_tot/h_w_httrans)

            # Coolant temp
            T_c_derived = T_wc - (q_tot/h_c_httrans)

            temp_resid = max(abs(T_wg-T_wall_hg), abs(T_wc-T_wall_c), abs(T_c_derived-T_c_0))
            if temp_resid < tol or j == max_iter:
                if j == max_iter:
                    # We are going to use these values if they don't converge in time
                    # We'll figure out the issues later
                    print(f"Slice {i} failed to converge!")
                T_c_0 = T_c_derived
                T_wall_c = T_wc
                T_wall_hg = T_wg
                return T_c_0, T_wall_c, T_wall_hg, P_c_0, T_aw_i
            else:
                # No convergence yet
                T_c_0 = T_c_derived
                T_wall_c = T_wc
                T_wall_hg = T_wg

    def other_station_function(T_c_0, T_wall_c, T_wall_hg, P_c, i):
        """
        This function handles all stations after the first (at the nozzle exit)
        This will handle the pressure drop and will use the ds between each station point
        """
        # Lets set the pressure to the incoming pressure
        # This is going to get updated after the pressure drop is calculated
        P_c_0 = P_c

        # First lets determine the pressure at this station
        # This is done by subtracting the pressure from the original pressure
        # Length of slice using just pythagorean
        ds = np.sqrt((y[i] - y[i + 1]) ** 2 + (x[i] - x[i + 1]) ** 2)
        Dh_c = 2 * depth[i] * width[i] / (depth[i] + width[i])
        for j in range(max_iter):
            # == COOLANT == #


            # Coolant Properties
            st.update(CP.PT_INPUTS, P_c, T_c_0)
            mu_c = st.viscosity()
            rho_c = st.rhomass()

            # Reynolds number
            Re_c = mdot * y[i] / (Dh_c * num_ch * mu_c)

            # Velocity
            v_c = mdot / (depth[i] * width[i] * rho_c * num_ch)

            if Re_c < 2300:
                f = 64 / Re_c

            # Transient
            elif 2300 < Re_c < 4000:
                # Treat conservatively
                # Do a blend of the two
                f_turb = Moody_plot(eps_dh, Re_c)
                f_lam = 64 / Re_c
                w = (Re_c - 2300) / (4000 - 2300)
                f = (1 - w) * f_lam + w * f_turb

            # Turbulent
            else:
                # From moody
                f = Moody_plot(eps_dh=eps_dh, Re=Re_c)

            # Darcy pressure drop
            dP_c = f * (ds / Dh_c) * (rho_c * v_c ** 2 / 2)

            # Now update the pressure at this slice
            # This is NOT an energy method
            # And this will not change
            P_c = P_c_0 - dP_c

            # Using the new coolant pressure, find the coolant props
            # Coolant temp is going to be iterated, so first iteration is the
            # T_c_0 from the previous step
            # but will be updated at the end of the iteration
            st.update(CP.PT_INPUTS, P_c, T_c_0)
            cp_c = st.cpmass()
            mu_c = st.viscosity()
            k_c = st.conductivity()
            rho_c = st.rhomass()

            # Compute the new coolant Reynolds number and Prandtl number
            Pr_c: float = cp_c * mu_c / k_c
            Re_c: float = rho_c * v_c * Dh_c / mu_c

            # Nusselt number using piecewise from _-_ (Dittus Boltzman form)
            if 0.1 <= Pr_c <= 1.0:
                Nu_c: float = 0.02155 * Re_c ** 0.8018 * Pr_c ** 0.7095
            elif 1.0 < Pr_c <= 3.0:
                Nu_c: float = 0.01253 * Re_c ** 0.8413 * Pr_c ** 0.6179
            else:
                Nu_c: float = 0.00881 * Re_c ** 0.8991 * Pr_c ** 0.3911

            h_c_httrans: float = Nu_c * k_c / Dh_c


            # == WALL == #
            T_w = 0.5 * (T_wall_c + T_wall_hg)
            # Update wall props with temperature if available
            Material_Properties(dic=data, T=T_w)
            k_wall: float = data["W"]["k"]
            h_w_httrans: float = t_wall / k_wall


            # == HOT GAS == #
            Pr_hg_0 = data["H"]["Pr"][i]
            recovery_hg = Pr_hg_0 ** (1 / 3)
            T_aw_hg: float = T_hg[i] + (recovery_hg * (U[i] ** 2 / (2 * cp_hg[i])))

            Re_hg = 4 * mdot_hg / (np.pi * 2*y_i * mu_hg[i])
            # h_hg_httrans: float = (0.026
            #                 * Re_hg**0.8
            #                 * Pr_hg[i]**0.6
            #                 * (Dt/R)**0.1
            #                 * (At/A[i])**0.9)
            #
            # sigma1 = (0.5 * (T_aw_hg / Tc) * (1+ (gamma_hg[i]-1) / 2 * M[i]**2) + 0.5) ** -0.68
            # sigma2 = (1 + (gamma_hg[i]-1) / 2 * M[i]**2) ** -0.12
            # h_hg_httrans *= sigma1 * sigma2

            # NASA hot gas heat transfer coefficient
            h_hg_httrans = 0.026 * (k_hg[i] / (2 * np.min(data["E"]["y"]))) * Re_hg ** 0.8 * Pr_hg_0 ** 0.3

            # Compute all the heat fluxes to make sure they match
            H = (1 / h_hg_httrans + 1 / h_w_httrans + 1 / h_c_httrans) ** -1
            q_tot = H * (T_aw_hg - T_c_0)

            # Hot gas side wall temp
            T_wg = T_aw_hg - (q_tot / h_hg_httrans)

            # Coolant side wall temp
            T_wc = T_wg - (q_tot / h_w_httrans)

            # Coolant temp
            T_c_derived = T_wc - (q_tot / h_c_httrans)

            temp_resid = max(abs(T_wg - T_wall_hg), abs(T_wc - T_wall_c), abs(T_c_derived - T_c_0))
            if temp_resid < tol or j == max_iter:
                if j == max_iter:
                    # We are going to use these values if they don't converge in time
                    # We'll figure out the issues later
                    print(f"Slice {i} failed to converge!")
                T_c_0 = T_c_derived
                T_wall_c = T_wc
                T_wall_hg = T_wg
                return T_c_0, T_wall_c, T_wall_hg, P_c, T_aw_hg
            else:
                # No convergence yet
                T_c_0 = T_c_derived
                T_wall_c = T_wc
                T_wall_hg = T_wg

    # ============ #
    # == SOLVER == #
    # ============ #
    energy_method = data["Solver"]["EnergyMethod"]

    FLUID_MAP = {
        "LOX": "Oxygen",
        "GOX": "Oxygen",
        "RP-1": "n-Dodecane",
        "Kerosene": "n-Dodecane",
        "Kero": "n-Dodecane",
        "CH4": "Methane"
    }

    # == Engine Geometry == #
    x = data["E"]["x"]
    y = data["E"]["y"]
    U = data["Flow"]["U"]
    M = data["Flow"]["M"]
    N = len(x)
    R = (data["E"]["r_exit"] + data["E"]["r_entry"]) / 2
    Dt = np.min(y)*2
    A = np.pi / 4 * y**2
    At = np.pi/4 * Dt**2

    # == Channel Geometry == #
    depth:  dict        = data["C"]["depth_arr"]
    width:  dict        = data["C"]["width_arr"]
    num_ch: float       = data["C"]["num_ch"]
    t_wall: float       = data["W"]["thickness"]
    k_wall: float       = data["W"]["k"]
    eps_dh: float       = data["W"]["roughness"]

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
    Pr_hg:      dict    = data["H"]["Pr"]

    # == Coolant Properties == #
    coolant:    str     = data["F"]["Type"]
    coolant:    str     = FLUID_MAP.get(coolant, coolant)
    mdot:       float   = data["F"]["mdot"]

    """These are initial conditions for the first slice ONLY
    All other slices will be set at the end of the previous slice
    These are the stagnation conditions however"""

    # == Hot Gas Properties == #
    H_hg_0: float = data["H"]["H"]
    C_hg: float   = 0.023


    # == Coolant Properties == #
    T_c_0:      float       = data["F"]["T"]
    P_c_0:      float       = data["F"]["P"]
    rho_c_0:    float       = data["F"]["rho"]
    H_c_0:      float       = data["F"]["H"]
    C_c:        float       = 1
    g_c:        float       = 1


    # == Wall Properties == #
    T_wall_hg:  float       = data["W"]["InitialTemp"]
    T_wall_c:   float       = data["W"]["InitialTemp"]

    """These are all the storage items to keep track of data"""
    # == Stagnation Storage == #
    H_c_0_arr:  np.ndarray  = np.zeros(N, dtype=float)

    # == Static Storage == #
    P_c_arr:    np.ndarray  = np.zeros(N, dtype=float)
    H_c_arr:    np.ndarray  = np.zeros(N, dtype=float)
    rho_c_arr:  np.ndarray  = np.zeros(N, dtype=float)
    Re_c_arr:   np.ndarray  = np.zeros(N, dtype=float)
    v_c_arr:    np.ndarray  = np.zeros(N, dtype=float)

    # == Temperature Storage == #
    T_wall_c_arr:   np.ndarray  = np.zeros(N, dtype=float)
    T_wall_hg_arr:  np.ndarray  = np.zeros(N, dtype=float)
    T_c_arr:        np.ndarray  = np.zeros(N, dtype=float)
    T_aw_arr:       np.ndarray = np.zeros(N, dtype=float)

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

        # Iterations take place within these functions
        if i == N-1:
            T_c_0, T_wall_c, T_wall_hg, P_c, T_aw = first_station_function(T_c_0=T_c_0, T_wall_hg=T_wall_hg,
                                                                     T_wall_c=T_wall_c, i=i)

        else:
            T_c_0, T_wall_c, T_wall_hg, P_c, T_aw = other_station_function(T_c_0=T_c_0, T_wall_hg=T_wall_hg,
                                                                     T_wall_c=T_wall_c, P_c=P_c, i=i)

        # Commit slice info to a data array
        T_c_arr[i] = T_c_0
        T_wall_c_arr[i] = T_wall_c
        T_wall_hg_arr[i] = T_wall_hg
        T_aw_arr[i] = T_aw
        P_c_arr[i] = P_c

    q = {"T_cool": T_c_arr,
         "T_wall_coolant": T_wall_c_arr,
         "T_wall_gas": T_wall_hg_arr,}

        # If first slice, static is equal to stagnation
        # Otherwise this will be set after convergence



        # Iterate multiple times to get the proper values
        # for j in range(max_iter):
        #
        #
        #
            # # == COOLANT == #
            #
            # # If not the first slice, and then if first iteration or not
            # if i != N-1:
            #     if j == 0:
            #         rho_c: float = float(rho_c_arr[i+1])
            #     else:
            #         # This is derived directly from the previous iteration, no manual adjusting of any kind
            #         rho_c: float = CP.PropsSI("rhomass", "P", P_c, "H", H_c)
            #
            # else:
            #     if j == 0:
            #         H_c: float = H_c_0
            #         P_c: float = P_c_0
            #         rho_c: float = rho_c_0
            #
            # # If first iteration, update stagnation enthalpy with eqn 9
            # if j == 0:
            #     H_c_0: float = H_c_0_arr[i+1] + (q_c_arr[j+1] * dx_i / mdot)
            #
            # else:
            #     H_c_0: float = H_c_0_arr[i+1] + ((q_c_arr[j] + q_c_arr[j+1]) * dx_i / mdot)
            #
            # # Coolant velocity
            # v_c: float = mdot / (rho_c * depth[i] * width[i] * num_ch)
            #
            # # Static enthalpy
            # H_c: float = H_c_0 - (v_c**2 / 2)
            #
            # # Coolant Reynolds values (static and reference)
            # mu_c: float = CP.PropsSI("viscosity", "P", P_c_arr[i + 1], "H", H_c_arr[i + 1])
            # Re_c: float = mdot * y_i / (depth[i] * width[i] * num_ch * mu_c)
            #
            # Re_c_Ref: float = Re_c * (rho_hg[i] / rho_c) * (mu_c / mu_hg[i])
            #
            # # Average coolant reynolds number for better accuracy
            # # Do not use if first slice
            # if j != N-1:
            #     Re_c_avg: float = 0.5 * (Re_c_arr[i] + Re_c_arr[i+1])
            #     Re_c_Ref_avg: float = 0.5 * (Re_c_Ref[i] + Re_c_Ref_arr[i+1])
            # else:
            #     Re_c_avg: float = Re_c
            #     Re_c_Ref_avg: float = Re_c_Ref
            #
            # # Friction factor using Moody diagram
            # C1: float = C_c / 0.023
            # if Re_c_Ref_avg < 2.2e3:
            #     f_c: float = 64 / Re_c_Ref_avg
            # elif 2.2e3 <= Re_c_Ref_avg < 10e4:
            #     f_c: float = 4*C1*(0.0014+ (0.125/Re_c_Ref_avg**0.32))
            # else:
            #     f_c: float = 0.778*C_c*Re_c_Ref_avg**-0.1021
            #
            # # Viscous Pressure drop (darcy)
            # dP_c_f: float = f_c/(4*g_c) * ((rho_c + rho_c_arr[i+1])/(y_i + y[i+1])) * (v_c**2 + v_c_arr[i+1]**2) * dx_i
            #
            # # Momentum pressure drop
            # dP_c_M: float = ((2 / ((N*depth[i+1]*width[i+1]) + (N*depth[i]*depth[i+1]))) *
            #           mdot**2/g_c *
            #           ((1 / (rho_c*depth[i]*width[i]*N)) - (1 / (rho_c_arr[i+1]*depth[i+1]*width[i+1]*N))))
            #
            # # New coolant pressure (eqn 15)
            # P_c: float = P_c_arr[i+1] - (dP_c_f + dP_c_M)
            #
            # # Updated coolant wall properties
            # st.update(CP.PT_INPUTS, P_c, T_wall_c)
            # cp_w_c: float = st.cpmass()
            # mu_w_c: float = st.viscosity()
            # k_w_c: float = st.conductivity()
            # P_c_w: float = st.p()
            # H_c_w: float = st.hmass()
            #
            # # Coolant reference enthalpy (eqn 16)
            # H_c_Ref: float = 0.5*(H_c + H_c_w) + 0.194*(H_c_0 - H_c)
            #
            # # Update coolant reference properties
            # st.update(CP.PSmass_INPUTS, P_c, H_c_Ref)
            # T_c_Ref: float = st.T()
            # cp_c_Ref: float = st.cpmass()
            # mu_c_Ref: float = st.viscosity()
            # k_c_Ref: float = st.conductivity()
            # P_c_Ref: float = st.p()
            # rho_c_Ref: float = st.rhomass()
            #
            # # Update coolant static properties
            # st.update(CP.PSmass_INPUTS, P_c, H_c)
            # T_c: float = st.T()
            # cp_c: float = st.cpmass()
            # mu_c: float = st.viscosity()
            # k_c: float = st.conductivity()
            # P_c: float = st.p()
            # rho_c: float = st.rhomass()
            #
            # # Coolant Prandtl Number reference
            # Pr_c_Ref: float = cp_c_Ref * mu_c_Ref / k_c_Ref
            #
            # # Coolant adiabatic wall enthalpy
            # H_c_aw: float = H_c + (Pr_c_Ref**(1/3) * (H_c_0 - H_c))
            #
            # # Coolant adiabatic wall temp
            # T_c_aw: float = CP.PropsSI("T", "P", P_c, "H", H_c_aw)
            #
            # # Coolant heat transfer using dittus-boltzman
            # d_hyd: float = 2 * (depth[i] * width[i]) / (depth[i] + width[i])
            # Re_c_Ref: float = rho_c_Ref * v_c * d_hyd / mu_c_Ref
            #
            # if 0.1 <= Pr_c_Ref <= 1.0:
            #     Nu_c_Ref: float = 0.02155 * Re_c_Ref**0.8018 * Pr_c_Ref**0.7095
            # elif 1.0 < Pr_c_Ref <= 3.0:
            #     Nu_c_Ref: float = 0.01253 * Re_c_Ref**0.8413 * Pr_c_Ref**0.6179
            # else:
            #     Nu_c_Ref: float = 0.00881 * Re_c_Ref**0.8991 * Pr_c_Ref**0.3911
            #
            # h_c_httrans: float = Nu_c_Ref * k_c_Ref / d_hyd
            #
            #
            # # == WALL == #
            # # Update wall properties, updated from previous iteration
            # Material_Properties(dic=data)
            # # Wall heat transfer coefficient
            # h_w_httrans: float = t_wall / k_wall
            #
            #
            # # == HOT GAS == #
            # H_wall_hg: float = H_hg[i] + cp_hg[i]*(T_wall_hg - T_hg[i])
            #
            # # Hot gas reference enthalpy
            # H_hg_Ref: float= 0.5 * (H_wall_hg + H_hg[i]) + 0.18 * (H_hg_0 - H_hg[i])  # Place in loop
            #
            # # Hot gas reference temperature
            # T_hg_Ref: float = CP.PropsSI("T", "P", P_hg, "H", H_hg_Ref)
            #
            # # Hot gas Prandtl reference number
            # Pr_hg_Ref: float = cp_hg[i]*mu_hg[i]/k_hg[i]
            #
            # # Hot gas adiabatic wall temp
            # H_hg_aw: float = H_hg[i] + (Pr_hg_Ref**(1/3) * (H_hg_0 - H_hg[i]))
            #
            # # Hot gas reynolds number reference
            # Re_hg_Ref: float = 4 * mdot_hg * T_hg[i] / (np.pi * 2*y_i * mu_hg[i] * T_hg_Ref)
            #
            # # Hot gas side heat transfer coefficient
            # h_hg_httrans: float = (C_hg * k_hg[i] / (y_i*2)) * Re_hg_Ref**0.8 * Pr_hg_Ref**0.3
            #
            # # == COMBINED HEAT FLUX == #
            #
            # # Basically a resistance setup
            # h_overall: float = ((1/h_hg_httrans) + h_w_httrans + (1/h_c_httrans))**-1
            #
            # # Total heat flux q = H*(Taw - Tco)
            # q_tot: float = h_overall * ((H_hg_aw/cp_hg[i]) - (H_c_Ref/cp_c_Ref))
            #
            # # New hot gas side wall temp
            # T_hg_w_new: float = (H_hg_aw / cp_hg[i]) - (q_tot/h_hg_httrans)
            # T_c_w_new: float = T_hg_w_new - (q_tot/h_w_httrans)
            # T_c_new: float = T_c_w_new - (q_tot/h_c_httrans)
            #
            #
            # # Now compare ALL the new temps with the previous iteraiton
            # err_hg_w: float = abs(T_hg_w_new - T_hg_w_old) / T_hg_w_new
            # err_c_w: float = abs(T_c_w_new - T_c_w_old) / T_c_w_new
            # err_c: float = abs(T_c_new - T_c_old) / T_c_new
            # if (err_hg_w or err_c or err_c_w) < tol:
            #     break
            #
            # # == UPDATE FOR NEXT ITERATION == #
            #
            # # Temps
            # T_hg_w_old: float = T_hg_w_new
            # T_c_w_old: float = T_c_w_new
            # T_c_old: float = T_c_new
            #
            # # Coolant properties
            # v_c_arr[i] = v_c
            # P_c_arr[i] = P_c
            # rho_c_arr[i] = rho_c
            # Re_c_arr[i] = Re_c
            # Re_c_Ref_arr[i] = Re_c_Ref




        # Next stations static point is the previous stations converged stagnation??




