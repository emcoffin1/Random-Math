import numpy as np
from NozzleFlow.GasProperties import Fluid_Properties

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
    h_f, Nu, Re = dittus_appro(dic=info, dimension=0, iteration=0)

    info["F"]["h"] = h_f
    info["F"]["Nu"] = Nu
    info["F"]["Re"] = Re

    mdot_f, cp_f, T_f = info["F"]["mdot"], info["F"]["cp"], info["F"]["T"]

    t_w, k_w = info["W"]["thickness"], info["W"]["k"]

    x, y, T, M = info["Flow"]["x"], info["Flow"]["y"], info["Flow"]["T"], info["Flow"]["M"]

    D = 2*y
    Dt = 2*Rt

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


def bartz_heat_transfer_1d(info: dict, max_iteration=10000, tol=1e-5):

    """Engine and Hot Gas Information"""
    # =========================== #
    # ==  Dictionary breakdown == #
    # =========================== #
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
    T_wall_coolant: np.ndarray  = np.zeros(N, dtype=float)
    T_wall_gas: np.ndarray      = np.zeros(N, dtype=float)
    T_c: np.ndarray             = np.zeros(N, dtype=float)
    Q_dot: np.ndarray           = np.zeros(N, dtype=float)
    R_hg_w_arr: np.ndarray      = np.zeros(N, dtype=float)
    R_w_c_arr: np.ndarray       = np.zeros(N, dtype=float)
    R_w_w_arr: np.ndarray       = np.zeros(N, dtype=float)

    """
    Initial Conditions
    Initial temperature is set as   
    """
    info["C"]["h"]              = np.zeros_like(x)
    info["C"]["Nu"]             = np.zeros_like(x)
    info["C"]["Re"]             = np.zeros_like(x)
    info["C"]["T"]              = np.zeros_like(x)

    T_coolant                   = float(T_f)

    """March along x"""
    # March through all slices, i
    # Starts from -1 and works backwards (-1, -2, -3, ...)
    # This is done as fuel (coolant) enters the engine from the nozzle exit side
    for i in range(N-1, -1, -1):

        progress = (N - 1 - i) / (N - 1) * 100
        print(f"\rCooling solver progress: {progress:5.1f}%", end="", flush=True)

        # Slice Radius
        y_i = float(y[i])
        # Slice thickness
        dx_i = float(dx_i_arr[i])

        if dx_i <= 0:
            raise ValueError(f"dx_i = {dx_i} must be positive!")

        T_slice_in = float(T_coolant)
        T_guess = T_slice_in

        # Last computed values for use after convergence
        Q_i = np.nan
        R_w_c = np.nan
        R_w_w = np.nan

        # Iterate a max of j times
        for j in range(max_iteration):

            info["C"]["T"][i] = float(T_guess)

            h_coolant, Nu, Re, A_fs, n_fs = dittus_appro(dx=dx_i, dic=info, dimension=1, step=i)

            info["C"]["h"][i] = h_coolant
            info["C"]["Nu"][i] = Nu
            info["C"]["Re"][i] = Re

            # Hot Gas properties
            Pr = (mu[i] * cp[i]) / k[i]
            r_rec = Pr ** (1/3)

            # Adiabatic wall temperature
            Taw_i = Tc * (1 + r_rec * (gamma[i]-1)/2 * M[i]**2) / (1 + (gamma[i]-1)/2 * M[i]**2)
            Taw[i] = Taw_i

            # Bartz convection coefficient approximation
            h_hg[i] = bartz_approx(Taw=Taw[i], dic=info, dimension=1, step=i, iteration=j)

            # Thermal resistance
            # Hot gas side
            R_hg_w = 1 / (2 * np.pi * y_i * dx_i * h_hg[i])

            # Wall conduction
            R_w_w = np.log((y_i+t_w) / y_i) / (2 * np.pi * dx_i * k_w)

            # Wall to coolant
            R_w_c = 1 / (n_fs * A_fs * h_coolant)

            # Total resistance network
            R_total = R_hg_w + R_w_w + R_w_c
            if R_total <= 0 or not np.isfinite(R_total):
                raise ValueError(f"Invalid total resistance at i={i}, j={j}: R_total={R_total}")

            # Heat flux to coolant
            Q_i = (Taw_i - T_slice_in) / R_total


            # Temperature adjustment
            T_coolant_out = T_slice_in + Q_i / (mdot_f * cp_f)

            T_coolant_new = (T_slice_in + T_coolant_out) / 2

            residual = abs(T_coolant_new - T_guess)
            if residual < tol:
                T_guess = T_coolant_new
                break

            # if not (Taw[i] > T_wall_gas[i] > T_wall_coolant[i] > T_c[i]):
            #     raise AssertionError(
            #         f"\nThermal ordering violated at slice {i}\n"
            #         f"Taw           = {Taw[i]:.2f} K\n"
            #         f"T_wall_gas    = {T_wall_gas[i]:.2f} K\n"
            #         f"T_wall_coolant= {T_wall_coolant[i]:.2f} K\n"
            #         f"T_coolant     = {T_c[i]:.2f} K\n"
            #         f"Check: Taw > Twg > Twc > Tc\n"
            #     )
            T_guess = T_coolant_new

        else:
            # Didnt converge
            raise RuntimeError(f"Coolant iteration failed to converge at i={i}. Last T_guess={T_guess:.4f}, Q_i={Q_i:.4f}")

        # Coolant and heat transfer slice results
        T_coolant = float(T_guess)
        T_c[i] = T_coolant
        Q_dot[i] = float(Q_i)

        # Resistance Results
        R_hg_w_arr[i] = R_hg_w
        R_w_w_arr[i] = R_w_w
        R_w_c_arr[i] = R_w_c

        # Wall temperature for slice
        T_wall_coolant[i] = T_coolant + Q_i * R_w_c
        T_wall_gas[i] = T_wall_coolant[i] + Q_i * R_w_w

        # Optional physical ordering check
        # if not (Taw[i] > T_wall_gas[i] > T_wall_coolant[i] > T_c[i]):
        #     raise RuntimeError(
        #         f"Thermal ordering failed at i={i}: "
        #         f"Taw={Taw[i]:.2f}, Twg={T_wall_gas[i]:.2f}, Twc={T_wall_coolant[i]:.2f}, Tc={T_c[i]:.2f}"
        #     )

    dic = {"hg": h_hg,
           "Q_dot": Q_dot,
           "T_aw": Taw,
           "T_cool": T_c,
           "T_wall_coolant": T_wall_coolant,
           "T_wall_gas": T_wall_gas,
           "R_hg_w": R_hg_w_arr,
           "R_w_w": R_w_w_arr,
           "R_w_c": R_w_c_arr,}


    print("\nCooling solution complete.")
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

    Dt = np.min(dic["E"]["y"]) * 2
    Pc = dic["E"]["Pc"]
    cstar = dic["H"]["cstar"]
    eps = dic["E"]["aspect_ratio"]
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
             * (Pc / cstar)**0.8 * eps**-0.9

        sigma1 = (0.5 * (Taw / Tc) * (1 + (gamma - 1)/2 * M**2) + 0.5)**-0.68
        sigma2 = (1 + (gamma - 1)/2 * M**2)**-0.12

        return h_hg * sigma1 * sigma2

    else:
        i = step
        mu = dic["H"]["mu"][i]
        cp = dic["H"]["cp"][i]
        gamma = dic["H"]["gamma"][i]
        Pr = dic["H"]["Pr"][1]
        Mi = M[i]


        h_hg = (0.026 / Dt**0.2) * (mu**0.2 * cp / Pr**0.6) \
             * (Pc / cstar)**0.8 * np.abs(eps[i])**-0.9

        if iteration != 0:
            # Add correction if iteration is not first pass
            sigma1 = (0.5 * (Taw / Tc) * (1 + (gamma - 1) / 2 * Mi ** 2) + 0.5) ** -0.68
            sigma2 = (1 + (gamma - 1) / 2 * Mi ** 2) ** -0.12

            h_hg = h_hg * sigma1 * sigma2

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
        type, thickness_w, spacing, height = (dic["C"]["Type"], dic["W"]["thickness"],
                                                     dic["C"]["spacing"], dic["C"]["height"])


    else:
        raise ValueError(f"Number of channels is not defined")

    # Compute hydraulic diameter depending on shape type
    if type == "Square":
        Ah = np.pi / num_ch * (height**2 + 2*y*height + 2*thickness_w*height) - (spacing*height)

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
            G = mdot / (Ah * num_ch)
            Re = G * Dh / mu

        else:
            raise ValueError("Need fuel mass flow rate and channel area (flow area) "
                             "to determine convective heat transfer coefficient")

        # Calculate Nu based on flow characteristics
        if Re < 2300:
            # Fully-developed laminar, constant wall heat flux
            Nu = 4.36

        else:
            # Gnielinksi (for smooth duct)
            f = (0.79 * np.log(Re) - 1.64) ** -2
            Nu = ((f / 8.0) * (Re - 1000) * Pr) / (1.0 + 12.7 * np.sqrt(f / 8) * (Pr ** (2 / 3) - 1.0))

        # h_f = Nu * k / Dh
        h_f = 0.023*Re**0.8*Pr**0.4*(k / Dh)


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
        # dic["C"]["Afs"] = A_fs
        # dic["C"]["n_fs"] = n_fs

        return h_f, Nu, Re, A_fs, n_fs


    elif type == "Circle":
        Dh = dic["C"]["width"]
        Ah = np.pi * (Dh/2)**2


