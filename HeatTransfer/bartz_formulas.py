import numpy as np
from NozzleFlow.GasProperties import Fluid_Properties, HotGas_Properties
import CoolProp.CoolProp as CP

"""https://www.sciencedirect.com/science/article/pii/S2214157X25008834"""

def bartz_heat_transfer_const(info: dict, w=0.7):
    """
    Uses Bartz Formula to compute the gas-side convective heat transfer coefficient
    Essentially shows you how efficiently the hot gas transfers heat to the wall.
    qdot is the heat flux per unit area, or how much heat is transferred per area of wall

    This function currently assumes a constant coolant temperature
    :param info: Dictionary of all things
    """
    # Dictionary breakdown
    Pc, Rt, gamma, R, k, Tc, mdot, mu, cp, k, cstar = (info["E"]["Pc"], info["E"]["Rt"], info["H"]["gamma"], info["H"]["R"],
                                                info["H"]["k"],
                                                info["E"]["Tc"], info["E"]["mdot"], info["H"]["mu"][1],
                                                info["H"]["cp"][1], info["H"]["k"][1], info["H"]["cstar"])
    h_f, Nu, Re = dittus_appro(dic=info, dimension=0, step=0, dx=0)

    info["F"]["h"] = h_f
    info["F"]["Nu"] = Nu
    info["F"]["Re"] = Re

    mdot_f, cp_f, T_f = info["F"]["mdot"], info["F"]["cp"], info["F"]["T"]

    t_w, k_w = info["W"]["thickness"], info["W"]["k"]

    x, y, T, M = info["Flow"]["x"], info["Flow"]["y"], info["Flow"]["T"], info["Flow"]["M"]


    for i in range(1):

        # Adiabatic wall temperature calculation
        Pr = info["H"]["Pr"][1]
        r = Pr ** (1 / 3)
        T0 = Tc  # chamber stagnation; assume constant
        Taw = T0 * (1 + r * (gamma - 1) / 2 * M ** 2) / (1 + (gamma - 1) / 2 * M ** 2)

        h_h = bartz_approx(Taw=Taw, dic=info, dimension=0, step=0, iteration=0)


        """Knowing hg, we can now solve for the heat flux
        We do this by looking at the thermal resistance through all stages
        In this section, we only care about gas side to wall conduction to ambient
       Through the wall, and then 
        Rc = 1/hc
        Rw = tw/kw
        R = Rg + Rw + Rc
        """

        # Rg = 1 / hg
        Rw = t_w / k_w
        Rf = 1 / h_f
        Rh = 1 / h_h
        R_total = Rf + Rw + Rh

        """
        Find the initial wall temperature
        Use initial wall temperature to determine the heat flux (q")
        """
        q_conv  = (Taw - T_f) / R_total
        # T_wall_initial = (T_f + h_h * (Rw + Rf) * Taw) / (1 + h_h * (Rw + Rf))

        # q" (heat flux) W/m^2
        # q_conv = h_h * (Taw - T_wall_initial)
        T_wall_initial = Taw - q_conv * Rh

        dic = {"hg": h_h, "qdot": q_conv, "T_wi": T_wall_initial, "T_aw": Taw, "T_cool": np.ones_like(T)*T_f}

    return dic


def bartz_heat_transfer_1d(info: dict, max_iteration=100, tol=1e-13):

    """Engine and Hot Gas Information"""
    # =========================== #
    # ==  Dictionary breakdown == #
    # =========================== #
    """Random Solver Controls"""
    film_cooling                = info["FilmCool"]
    energy_method               = info["EnergyMethod"]

    HotGas_Properties(dic=info)
    """Engine and Hot gas information"""
    Tc                          = float(info["E"]["Tc"])
    gamma                       = info["H"]["gamma"]
    mu                          = info["H"]["mu"]
    k                           = info["H"]["k"]
    cp                          = info["H"]["cp"]

    """Coolant (fuel) bulk properties"""
    mdot_f                      = float(info["F"]["mdot"])
    cp_f                        = float(info["F"]["cp"])
    T_f                         = float(info["F"]["T"])  # inlet coolant temperature at nozzle-exit side

    """Wall Properties"""
    t_w                         = float(info["W"]["thickness"])
    k_w                         = float(info["W"]["k"])

    """Geometry and flow arrays"""
    x: list                     = info["E"]["x"]
    y: list                     = info["E"]["y"]
    M: list                     = info["Flow"]["M"]
    N = len(x)  # Number of x points

    """dx setup"""
    dx_seg                      = np.diff(x)
    dx_i_arr                    = np.empty(N, dtype=float)
    dx_i_arr[:-1]               = dx_seg
    dx_i_arr[-1]                = dx_seg[-1]


    """Storage"""
    Taw: np.ndarray             = np.zeros(N, dtype=float)
    h_hg: np.ndarray            = np.zeros(N, dtype=float)
    h_wc: np.ndarray            = np.zeros(N, dtype=float)
    T_wall_coolant: np.ndarray  = np.zeros(N, dtype=float)
    T_wall_gas: np.ndarray      = np.zeros(N, dtype=float)
    T_c_out: np.ndarray         = np.zeros(N, dtype=float)
    Q_dot: np.ndarray           = np.zeros(N, dtype=float)
    R_hg_w_arr: np.ndarray      = np.zeros(N, dtype=float)
    R_w_c_arr: np.ndarray       = np.zeros(N, dtype=float)
    R_w_w_arr: np.ndarray       = np.zeros(N, dtype=float)
    Re_arr: np.ndarray          = np.zeros(N, dtype=float)
    v_arr: np.ndarray           = np.zeros(N, dtype=float)
    Dh_arr: np.ndarray          = np.zeros(N, dtype=float)

    """
    Initial Conditions
    Initial temperature is set as   
    """
    info["C"]["h"]              = np.zeros_like(x)
    info["C"]["Nu"]             = np.zeros_like(x)
    info["C"]["Re"]             = np.zeros_like(x)
    info["C"]["T"]              = np.zeros_like(x)

    T_coolant                   = float(T_f)

    converge_count = 0

    """March along x"""
    # March through all slices, i
    # Starts from -1 and works backwards (-1, -2, -3, ...)
    # This is done as fuel (coolant) enters the engine from the nozzle exit side
    for i in range(N-1, -1, -1):

        progress        = (N - 1 - i) / (N - 1) * 100
        print(f"\rCooling solver progress: {progress:5.1f}%", end="", flush=True)

        # Slice Radius
        y_i             = float(y[i])
        # Slice thickness
        dx_i            = float(dx_i_arr[i])
        if dx_i <= 0:
            raise ValueError(f"dx_i = {dx_i} must be positive!")

        # Coolant inlet state for this slice
        h_slice_in      = float(info["F"]["H"])
        T_slice_in      = float(T_coolant)

        # Pressure handling
        P_slice_in = float(info["F"]["P"])
        P_slice_out = float(info["F"]["P"]) # needs to be replaced once dP modeled

        # Wall iteration guess
        T_guess = T_slice_in

        # Last computed values for use after convergence
        Q_i             = np.nan
        R_w_c           = np.nan
        R_w_w           = np.nan
        R_hg_w          = np.nan
        h_coolant       = np.nan
        Nu = Re = Dh = Ah = V = n_ef = np.nan
        T_coolant_out   = np.nan
        T_coolant_avg   = np.nan
        h_coolant_out   = np.nan

        # Initialize variables for first round
        if energy_method:
            info["F"]["H"] = h_slice_in
            info["F"]["P"] = P_slice_in

        else:
            info["F"]["T"] = T_slice_in
            info["F"]["P"] = P_slice_in

        Fluid_Properties(dic=info, coolant_only=True)

        # ==================================== #
        # == COOLANT TEMPERATURE ITERATIONS == #
        # ==================================== #

        # Iterate a max of j times
        for j in range(max_iteration):

            h_coolant, Nu, Re, Dh, Ah, V, n_ef = dittus_appro(dx=dx_i, dic=info, dimension=1, step=i)

            info["C"]["h"][i]   = h_coolant
            info["C"]["Nu"][i]  = Nu
            info["C"]["Re"][i]  = Re

            # Hot Gas properties
            Pr                  = (mu[i] * cp[i]) / k[i]
            r_rec               = Pr ** (1/3)

            # Adiabatic wall temperature
            Taw_i               = Tc * (1 + r_rec * (gamma[i]-1)/2 * M[i]**2) / (1 + (gamma[i]-1)/2 * M[i]**2)
            Taw[i]              = Taw_i

            # Bartz convection coefficient approximation
            h_hg_i              = bartz_approx(Taw=Taw[i], dic=info, dimension=1, step=i, iteration=j)
            h_hg[i]             = h_hg_i

            # Thermal resistance network
            # Hot gas side
            R_hg_w              = 1 / (2 * np.pi * y_i * dx_i * h_hg_i)

            # Wall conduction
            R_w_w               = np.log((y_i+t_w) / y_i) / (2 * np.pi * dx_i * k_w)

            # Wall to coolant
            R_w_c               = 1/ (h_coolant * Ah * n_ef)

            # Total resistance network
            R_total             = R_hg_w + R_w_w + R_w_c
            if R_total <= 0 or not np.isfinite(R_total):
                raise ValueError(f"Invalid total resistance at i={i}, j={j}: R_total={R_total}")

            # Heat flux to coolant
            Q_i                 = (Taw_i - T_guess) / R_total

            # Temperature adjustment
            if not energy_method:
                # Temperature marching
                cp_bulk          = float(info["F"]["cp"])
                T_coolant_out    = T_slice_in + Q_i / (mdot_f * cp_bulk)
                T_coolant_avg    = (T_slice_in + T_coolant_out) / 2

            else:
                # Enthalpy marching
                h_coolant_out    = h_slice_in + Q_i / mdot_f
                info["F"]["H"] = h_coolant_out
                info["F"]["P"] = P_slice_out

                Fluid_Properties(dic=info, coolant_only=True)
                T_coolant_out    = info["F"]["T"]
                T_coolant_avg    = 0.5 * (T_slice_in + T_coolant_out)


            residual             = abs(T_coolant_avg - T_guess)
            if residual < tol:
                converge_count += 1
                break

            T_guess             = T_coolant_avg


            # Update values for next iteration
            if energy_method:
                P_bulk = 0.5 * (P_slice_in + P_slice_out)
                h_bulk = 0.5 * (h_slice_in + h_coolant_out)
                info["F"]["P"] = P_bulk
                info["F"]["H"] = h_bulk
                Fluid_Properties(dic=info, coolant_only=True)

            else:
                P_bulk = 0.5 * (P_slice_in + P_slice_out)
                T_bulk = float(T_coolant_avg)
                info["F"]["P"] = P_bulk
                info["F"]["T"] = T_bulk
                Fluid_Properties(dic=info, coolant_only=True)

        # ========================== #
        # == COMMIT SLICE RESULTS == #
        # ========================== #

        T_coolant = float(T_coolant_out)
        T_c_out[i] = T_coolant_out
        Q_dot[i] = float(Q_i)

        # Random fluid values
        h_wc[i] = h_coolant
        Re_arr[i] = Re
        v_arr[i] = V
        Dh_arr[i] = Dh

        # Resistance Results
        R_hg_w_arr[i] = R_hg_w
        R_w_w_arr[i] = R_w_w
        R_w_c_arr[i] = R_w_c

        # Wall temperature for slice
        T_wall_coolant[i] = T_coolant_out + Q_i * R_w_c
        T_wall_gas[i] = T_wall_coolant[i] + Q_i * R_w_w

        # Advance coolant enthalpy for next slice
        if energy_method:
            info["F"]["H"] = float(h_coolant_out)


    Q_total = np.sum(Q_dot)
    dT_bulk = T_c_out[0] - T_c_out[-1]
    Q_from_bulk = mdot_f * cp_f * dT_bulk
    # print(f"\nConservations: {((Q_total - Q_from_bulk)/Q_from_bulk*100):.2f} %")

    dic = {"h_hg": h_hg,
           "h_wc": h_wc,
           "Q_dot": Q_dot,
           "T_aw": Taw,
           "T_cool": T_c_out,
           "T_wall_coolant": T_wall_coolant,
           "T_wall_gas": T_wall_gas,
           "R_hg_w": R_hg_w_arr,
           "R_w_w": R_w_w_arr,
           "R_w_c": R_w_c_arr,
           "v": v_arr,
           "Re": Re_arr,
           "Dh": Dh_arr,}


    print("\nCooling solution complete.", end="", flush=True)
    print()
    # print(f"{(converge_count/N*100):.2f}% of slices converged")
    return dic

def bartz_approx(Taw, dic:dict, dimension: int, step: int, iteration: int):
    """
    Hot gas convection coefficient approximation, with or without correction
    :param Taw: adiabatic wall temp
    :param dic: dictionary of all information
    :param dimension: Complexity of solver, 0 for simple automatically applies correction
    :param step: step through engine (x position for iterative process)
    :param iteration: Iteration count (used only for >0 dimension solver)
    :return: hot gas convection coefficient
    """

    Dt = dic["E"]["r_throat"] * 2
    Pc = dic["E"]["Pc"]
    cstar = dic["H"]["cstar"]
    eps = dic["E"]["aspect_ratio"]
    if np.any(eps <= 0):
        raise ValueError(f"There is a negative value in the aspect ratio. {eps}")
    # eps = dic["Flow"]["eps"]
    Tc = dic["E"]["Tc"]

    M = dic["Flow"]["M"]

    # Solver based on dimension
    if dimension == 0:
        # Throat-referenced properties
        mu = dic["H"]["mu"][1]
        cp = dic["H"]["cp"][1]
        gamma = dic["H"]["gamma"][1]
        Pr = dic["H"]["Pr"][1]

        h_hg = (0.026 / Dt**0.2) * (mu**0.2 * cp / Pr**0.6) \
             * (Pc / cstar)**0.8 * (1/eps)**0.9

        sigma1 = (0.5 * (Taw / Tc) * (1 + (gamma - 1)/2 * M**2) + 0.5)**-0.68
        sigma2 = (1 + (gamma - 1)/2 * M**2)**-0.12

        return h_hg * sigma1 * sigma2

    else:
        i  = step
        mu = dic["H"]["mu"][i]
        k  = dic["H"]["k"][i]
        gamma = dic["H"]["gamma"][i]
        Pr = dic["H"]["Pr"][1]
        Mi = M[i]
        D  = dic["E"]["y"][i] * 2
        R  = (dic["E"]["r_exit"] + dic["E"]["r_entry"]) / 2

        A = np.pi / 4 * D**2
        At = np.pi / 4 * Dt**2

        G  = Pc/cstar
        Re = G*D/mu

        Nu = (0.026
                * Re**0.8
                * Pr**0.6
                * (Dt/R)**0.1
                * (At/A)**0.9)

        sigma1 = (0.5 * (Taw / Tc) * (1 + (gamma - 1) / 2 * Mi ** 2) + 0.5) ** -0.68
        sigma2 = (1 + (gamma - 1) / 2 * Mi ** 2) ** -0.12

        h_hg = Nu * k / D *sigma1 * sigma2

        return h_hg



def dittus_appro(dx:float, dic:dict, dimension: int, step: int):
    """
    Computes convective heat transfer coefficient for the coolant in channels
    :param dic: dictionary of all information
    :param dimension: Complexity of solver, 0 for simple solves all points at once
    :param step: position in engine
    """
    i = step

    # Update some fluid properties if necessary
    # Only for 0 dimensional/constant coolant temp
    if dimension != 0:
        Fluid_Properties(dic=dic, coolant_only=True)


    rho, mu, k, cp, Pr, mdot = (dic["F"]["rho"], dic["F"]["mu"], dic["F"]["k"], dic["F"]["cp"],
                                dic["F"]["Pr"], dic["F"]["mdot"])


    y = dic["E"]["y"][i]
    k_wall = dic["W"]["k"]
    # Run the max number of channels if not already computed
    if dic["C"]["num_ch"] is not None:
        num_ch = dic["C"]["num_ch"]
        type, thickness_w, spacing, height = (dic["C"]["Type"].lower(), dic["W"]["thickness"],
                                                     dic["C"]["spacing"], dic["C"]["height"])


    else:
        raise ValueError(f"Number of channels is not defined")

    # Compute hydraulic diameter depending on shape type
    if type == "square":
        chan_depth = dic["C"]["depth_arr"][i]
        chan_width = dic["C"]["width_arr"][i]
        A = chan_width * chan_depth
        mdot_ch = mdot/num_ch

        # GEOMETRY #
        # Ph = 2 * (chan_depth + chan_width)
        # # OR USE THIS
        # Ph = chan_width
        # # OR
        Ph = chan_width + 2*chan_depth

        Dh = 4 * A / Ph
        A_cyl = 2 * np.pi * y * dx     # Area of cylinder now
        n_ef = min(1.0, num_ch* Ph/(2 * np.pi * y))

        # Velocity from mass flow rate equation
        V = mdot_ch / (rho * A)

        # Reynolds number using hydaulic diameter
        Re = mdot_ch * Dh / (mu * A)

        # Prandtl Number
        Pr = cp * mu / k

        # Nusselt
        Nu = 0.023 * Re**0.8 * Pr**0.4

        # Heat Transfer coefficient
        h_f = Nu * k / Dh

        return h_f, Nu, Re, Dh, A_cyl, V, n_ef


    elif type == "Square":
        a = dic["C"]["height"]
        Ah = np.pi / num_ch * (height**2 + 2*y*height + 2*thickness_w*height) - (spacing*height)
        A = a**2

        # Local exposed fin area
        A_f = 2 * height * dx

        # Total area exposed to coolant
        A_fs = (num_ch * A_f) + (dx * (2 * np.pi * (y + thickness_w) - (num_ch * spacing)))

        # Wetted perimeter
        P_wet = 2 * ((np.pi / num_ch * (2*y + 2*thickness_w + height)) - spacing + height)

        # Hydraulic Diameter
        Dh = 4 * Ah / P_wet

        if Dh <= 0:
            raise ValueError(f"Hydraulic Diameter must be positive: {Dh:.4f}, Ah: {Ah:.4f}, P_wet: {P_wet:.4f}")

        # Calculate reynolds using mass flux to avoid using velocity
        if mdot is not None and Ah is not None:
            # G = mdot / (Ah * num_ch)
            # Re = G * Dh / mu
            v = mdot/ num_ch / rho / Ah
            Re = rho * v * Dh / mu

        else:
            raise ValueError("Need fuel mass flow rate and channel area (flow area) "
                             "to determine convective heat transfer coefficient")

        Nu_lam = 4.36

        # Turbulent Gnielinksi
        f = (0.79*np.log(Re) - 1.64)**-2
        Nu_turb = ((f / 8.0) * (Re - 1000) * Pr) / (1.0 + 12.7 * np.sqrt(f / 8) * (Pr ** (2 / 3) - 1.0))

        Re1, Re2 = 2000, 5000
        w = (Re - Re1) / (Re2 - Re1)
        w = max(0.0, min(1.0, w))
        Nu = (1-w)* Nu_lam + w*Nu_turb

        h_f = Nu * k / Dh
        # h_f = 0.023*Re**0.8*Pr**0.4*(k / Dh)


        # Single fin efficiency
        if spacing <= 0 or not np.isfinite(spacing):
            raise ValueError(f"Invalid fin thickness (spacing): {spacing}")

        if k_wall <= 0 or not np.isfinite(k_wall):
            raise ValueError(f"Invalid wall conductivity: {k_wall}")

        if h_f < 0 or not np.isfinite(h_f):
            raise ValueError(f"Invalid coolant HTC: {h_f}")

        m = np.sqrt(2 * h_f / k_wall / spacing)
        n_f = np.tanh(m * height) / (m * height)

        # Finset efficiency
        n_fs = 1 - (num_ch * A_f / A_fs) * (1 - n_f)
        return h_f, Nu, Re, A_fs, n_fs, v, Dh, h_f


    elif type == "circle":
        Dh = dic["C"]["width"]
        Ah = np.pi * (Dh/2)**2



