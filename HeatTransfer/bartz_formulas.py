import numpy as np
from NozzleFlow.GasProperties import Fluid_Properties, HotGas_Properties, Material_Properties
import CoolProp.CoolProp as CP
from NozzleFlow._extra_utils import Moody_plot
from CoolProp.CoolProp import AbstractState


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
    film_cooling                = info["Solver"]["FilmCool"]
    energy_method               = info["Solver"]["EnergyMethod"]

    HotGas_Properties(dic=info)
    """Engine and Hot gas information"""
    Tc                          = float(info["E"]["Tc"])
    T_hg                            = float(info["Flow"]["T"])
    gamma                       = info["H"]["gamma"]
    mu                          = info["H"]["mu"]
    k                           = info["H"]["k"]
    cp                          = info["H"]["cp"]

    """Coolant (fuel) bulk properties"""
    mdot_f                      = float(info["F"]["mdot"])
    cp_f                        = float(info["F"]["cp"])
    T_f                         = float(info["F"]["T"])  # inlet coolant temperature at nozzle-exit side
    P_inlet                     = info["F"]["P"]

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
    P_bulk_arr: np.ndarray = np.zeros(N, dtype=float)
    info["F"]["rho_arr"]        = np.zeros(N, dtype=float)

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

    dP = info["C"]["dP_arr"]
    P_nodes = np.zeros(N)
    P_nodes[0] = P_inlet
    for o in range(1,N):
        P_nodes[o] = P_nodes[o-1] + dP[o-1]

    info["F"]["P_arr"] = P_nodes

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
        # Uses the previous stations value UNLESS first slice
        # In which stagnation conditions are used
        h_slice_in      = float(info["F"]["H"])
        T_slice_in      = float(T_coolant)

        # Pressure handling
        P_slice_in = np.sum(info["C"]["dP_arr"][:i]) + info["Injector"]["dP"]
        P_slice_out = np.sum(info["C"]["dP_arr"][:i-1]) + info["Injector"]["dP"]

        # Last computed values for use after convergence
        Q_i             = 0.0
        R_w_c           = np.nan
        R_w_w           = np.nan
        R_hg_w          = np.nan
        h_coolant       = np.nan
        Nu = Re = Dh = Ah = V = n_ef = np.nan
        T_coolant_out   = T_slice_in
        T_coolant_avg   = np.nan
        h_coolant_out   = np.nan

        # Initialize variables for first round
        if energy_method:
            info["F"]["H"] = h_slice_in
            info["F"]["P"] = P_slice_in

        else:
            info["F"]["T"] = T_slice_in
            info["F"]["P"] = P_slice_in

        # Get stagnation conditions
        if i == N-1:
            Fluid_Properties(dic=info, coolant_only=True)

        # ==================================== #
        # == COOLANT TEMPERATURE ITERATIONS == #
        # ==================================== #

        # Iterate a max of j times
        for j in range(max_iteration):

            h_coolant, Nu, Re, Dh, Ah, V, n_ef = dittus_appro(dx=dx_i, dic=info, dimension=1, step=i)

            # Hot Gas properties
            Pr                  = (mu[i] * cp[i]) / k[i]
            r_rec               = Pr ** (1/3)

            # Adiabatic wall temperature
            Taw_i               = T_hg[i] * (1 + r_rec * (gamma[i]-1)/2 * M[i]**2)
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
                raise ValueError(f"Invalid total resistance at i={i}, j={j}: R_total={R_total}\n"
                                 f"R_hw_w: {R_hg_w:.2f}\n"
                                 f"R_w_w: {R_w_w:.2f}\n"
                                 f"R_w_c: {R_w_c:.2f}\n"
                                 f"{h_coolant:.2f}\n"
                                 f"{Ah:.2f}\n"
                                 f"{n_ef:.2f}\n")

            P_bulk = 0.5 * (P_slice_in + P_slice_out)

            if energy_method:
                # Energy method to get coolant out temp
                h_coolant_out = h_slice_in + Q_i / mdot_f

                # Convert that to temp
                info["F"]["H"] = h_coolant_out
                info["F"]["P"] = P_bulk
                Fluid_Properties(dic=info, coolant_only=True)
                T_coolant_out = info["F"]["T"]
                cp_bulk = float(info["F"]["cp"])

            else:
                # Temp method to get coolant out temp
                cp_bulk = float(info["F"]["cp"])
                T_coolant_out    = T_slice_in + Q_i / (mdot_f * cp_bulk)
                info["F"]["T"] = T_coolant_out
                info["F"]["P"] = P_bulk
                Fluid_Properties(dic=info, coolant_only=True)

            T_wall = T_coolant_out + Q_i * R_w_c
            Q_new = (Taw_i - T_wall) / R_total

            residual = abs(Q_new - Q_i)
            Q_i = Q_new


            if residual < tol:
                break


        # ========================== #
        # == COMMIT SLICE RESULTS == #
        # ========================== #
        T_c_out[i] = T_coolant_out
        Q_dot[i] = float(Q_i)
        P_bulk_arr[i] = P_bulk

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

        info["F"]["rho_arr"][i] = info["F"]["rho"]

        # Advance coolant enthalpy for next slice
        if energy_method:
            info["F"]["H"] = float(h_coolant_out)

        else:
            info["F"]["T"] = T_coolant_out
        T_coolant = info["F"]["T"]


    Q_total = np.sum(Q_dot)
    dT_bulk = T_c_out[0] - T_c_out[-1]
    Q_from_bulk = mdot_f * cp_f * dT_bulk

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
           "Dh": Dh_arr,
           "P_c": P_bulk_arr}

    # conserv = (Q_total - Q_from_bulk)/Q_from_bulk*100
    # print(f"\nConservations: {((Q_total - Q_from_bulk)/Q_from_bulk*100):.2f} %")
    # print(f"\nCooling solution complete with {conserv:.2f}% conservation.", end="", flush=True)
    print(f"\nCooling solution complete.", end="", flush=True)
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

        h_hg = Nu * k / D * sigma1 * sigma2

        return h_hg



def dittus_appro(dx:float, dic:dict, dimension: int, step: int):
    """
    Computes convective heat transfer coefficient for the coolant in channels
    :param dic: dictionary of all information
    :param dimension: Complexity of solver, 0 for simple solves all points at once
    :param step: position in engine
    """
    i = step

    rho, mu, k, cp, Pr, mdot = (dic["F"]["rho"], dic["F"]["mu"], dic["F"]["k"], dic["F"]["cp"],
                                dic["F"]["Pr"], dic["F"]["mdot"])

    # if i == len(dic["E"]["x"]):
    #     rho = rho * (1+)

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

        if np.isnan(h_f):
            raise ValueError(f"h_f is not defined\n"
                             f"Nu = {Nu}\n"
                             f"k = {k}\n"
                             f"Dh = {Dh}\n"
                             f"Re = {Re}\n"
                             f"cp = {cp}\n"
                             f"mu = {mu}\n"
                             f"Pr = {Pr}\n")


        return h_f, Nu, Re, Dh, A_cyl, V, n_ef

    elif type == "circle":
        Dh = dic["C"]["width"]
        Ah = np.pi * (Dh/2)**2



def pressure_drop_assessment(data: dict):
    """
    This function requires a FULL coolant passage geometry sweep
    Will perform a simple darcy-weisbach sweep to determine the pressure drop
    through only the regen channels. This will not assess the minor losses.

    Assumptions for first pass:
        1. constant temperature->density/viscosity
    """

    # == STORAGE == #
    mdot        = data["F"]["mdot"] / data["C"]["num_ch"]
    rho         = data["F"]["rho"]
    rho_arr     = data["F"]["rho_arr"]
    if len(rho_arr) != 0:
        rho     = rho_arr

    mu          = data["F"]["mu"]
    depth       = data["C"]["depth_arr"]
    width       = data["C"]["width_arr"]
    eps         = 1e-5
    dx          = data["E"]["dx"]

    # == Geometric Characteristics == #
    Pw          = 2 * (depth + width)
    A           = depth * width
    Dh          = 4 * A / Pw

    # == Flow Characteristics == #
    V           = mdot / (rho * A)
    Re          = rho * V * Dh / mu
    eps_dh      = eps / Dh

    f = np.zeros_like(Re)
    for i in range(len(Re)):
        # Laminar
        if Re[i] < 2300:
            f[i] = 64 / Re[i]

        # Transient
        elif 2300 < Re[i] < 4000:
            # Treat conservatively
            # Do a blend of the two
            f_turb = Moody_plot(eps_dh[i], Re[i])
            f_lam = 64/Re[i]
            w = (Re[i] - 2300) / (4000 - 2300)
            f[i] = (1 - w)*f_lam + w*f_turb

        # Turbulent
        else:
            # From moody
            f[i] = Moody_plot(eps_dh=eps_dh[i], Re=Re[i])

    dP_array    = f * (dx/Dh) * (rho * V**2 / 2)
    sum_array   = dP_array.sum()

    # Full channel pressure difference array
    data["C"]["dP_arr"] = dP_array
    # Full channel pressure drop
    data["C"]["dP"]     = sum_array

    # Sets total pressure
    data["F"]["P"] = data["Injector"]["dP"] + dP_array.sum()

    # Saves the inlet pressure of the coolant
    data["F"]["StartingPressure"] = data["F"]["P"]

    if data["Solver"]["EnergyMethod"]:
        data["Solver"]["EnergyMethod"] = False
        Fluid_Properties(dic=data)
        data["Solver"]["EnergyMethod"] = True


def heat_transfer_solver(data: dict, max_iter=50, tol=1e-4):
    def first_station_function(T_c_0, T_wall_c, T_wall_hg, i):
        """
        This function is designed to handle all iterations for the first slice
        This is done since the first iteration doesn't have any previous stations to reference
        To handle this, all conditions are considered to be the initial static condition
        This will also be handled by assuming that x=N is the initial static conditions,
        and we'll assume zero loss through this region
        """
        ds = np.sqrt((y[i-1] - y[i]) ** 2 + (x[i-1] - x[i]) ** 2)
        T_c_in = T_c_0
        for j in range(max_iter):
            # == COOLANT == #

            if j == 0:
                T_c_out_guess = T_c_in
            T_c_mean = 0.5 * (T_c_out_guess + T_c_in)

            # Geometric Properties
            Dh_c = 2 * depth[i] * width[i] / (depth[i] + width[i])

            # First pull the initial static conditions of the coolant
            st.update(CP.PT_INPUTS, P_c_0, T_c_mean)
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

            # Cooling channel wetted perimeter
            Ph = 1*width[i] + 2*depth[i]

            # === WALL === #
            T_avg = 0.5 * (T_wall_c + T_wall_hg)
            Material_Properties(dic=data, T=T_avg)

            # == HOT GAS == #
            Pr_hg: float = mu_hg[i] * cp_hg[i] / k_hg[i]
            recovery_hg = Pr_hg ** (1 / 3)
            T_aw_i = T_hg[i] * (1 + (recovery_hg * (gamma_hg[i] - 1) / 2 * M[i] ** 2))

            Re_hg = rho_hg[i] * U[i] * y[i] * 2 / mu_hg[i]

            # Bartz correlation
            Nu: float = (0.026
                            * Re_hg**0.8
                            * Pr_hg**0.6
                            * (Dt/R)**0.1
                            * (At/A[i])**0.9)

            sigma1 = (0.5 * (T_aw_i / Tc) * (1+ (gamma_hg[i]-1) / 2 * M[i]**2) + 0.5) ** -0.68
            sigma2 = (1 + (gamma_hg[i]-1) / 2 * M[i]**2) ** -0.12
            h_hg_httrans = Nu * k_hg[i] / (y[i]*2) * sigma1 * sigma2

            # h_hg_httrans = 0.026 * (k_hg[i] / (2 * np.min(data["E"]["y"]))) * Re_hg ** 0.8 * Pr_hg_0 ** 0.3

            # == RESISTANCE NETWORK == #
            R_w_hg = 1 / (2 * np.pi * y[i] * h_hg_httrans * ds)
            R_w = np.log((y[i] + t_wall) / y[i]) / (2 * np.pi * k_wall * ds)
            R_w_c = 1 / (h_c_httrans * Ph * ds * num_ch)

            # Compute all the heat fluxes to make sure they match
            R_tot = R_w_hg + R_w + R_w_c
            q_tot = (T_aw_i - T_c_0) / R_tot

            # == TEMPERATURE DERIVATION == #
            # Hot gas side wall temp
            T_wg = T_aw_i - q_tot * R_w_hg

            # Coolant side wall temp
            T_wc = T_wg - q_tot * R_w

            # Coolant temp
            mdot_total = mdot
            T_c_out = T_c_in + q_tot / (mdot_total * cp_c_0)

            temp_resid = max(abs(T_wg - T_wall_hg), abs(T_wc - T_wall_c), abs(T_c_out - T_c_out_guess))/ T_aw_i
            if temp_resid < tol:
                return T_c_0, T_wall_c, T_wall_hg, P_c_0, T_aw_i, h_hg_httrans
            else:
                # No convergence yet
                T_c_out_guess = T_c_out
                T_wall_c = T_wc
                T_wall_hg = T_wg

        print(f"Slice {i} failed to converge!")
        return T_c_0, T_wall_c, T_wall_hg, P_c, T_aw_i, h_hg_httrans

    def other_station_function(T_c_0, T_wall_c, T_wall_hg, P_c, i):
        """
        This function handles all stations after the first (at the nozzle exit)
        This will handle the pressure drop and will use the ds between each station point
        """
        # Lets set the pressure to the incoming pressure
        # This is going to get updated after the pressure drop is calculated
        P_c_0 = P_c
        T_c_in = T_c_0

        # First lets determine the pressure at this station
        # This is done by subtracting the pressure from the original pressure
        # Length of slice using just pythagorean
        ds = np.sqrt((y[i] - y[i + 1]) ** 2 + (x[i] - x[i + 1]) ** 2)
        Dh_c = 2 * depth[i] * width[i] / (depth[i] + width[i])

        for j in range(max_iter):
            # == COOLANT == #

            if j == 0:
                T_c_out_guess = T_c_in
            T_c_mean = 0.5 * (T_c_in + T_c_out_guess)

            # Coolant Properties
            st.update(CP.PT_INPUTS, P_c, T_c_mean)
            mu_c = st.viscosity()
            rho_c = st.rhomass()

            # Velocity
            v_c = mdot / (depth[i] * width[i] * rho_c * num_ch)

            # Reynolds number
            Re_c = rho_c * v_c * Dh_c / mu_c

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
            try:
                st.update(CP.PT_INPUTS, P_c, T_c_mean)
            except Exception:
                print(f"Other station function error: P_c: {P_c} :: T_c_0: {T_c_0}")
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

            # Fin efficiency
            Ph = 1*width[i] + 2*depth[i]

            # == WALL == #
            T_w = 0.5 * (T_wall_c + T_wall_hg)
            # Update wall props with temperature if available
            Material_Properties(dic=data, T=T_w)
            k_wall: float = data["W"]["k"]
            h_w_httrans: float = k_wall / t_wall

            # == HOT GAS == #
            Pr_hg: float = mu_hg[i] * cp_hg[i] / k_hg[i]
            recovery_hg = Pr_hg ** (1 / 3)
            T_aw_hg = T_hg[i] * (1 + (recovery_hg * (gamma_hg[i] - 1) / 2 * M[i] ** 2))

            # Re_hg = 4 * mdot_hg / (np.pi * 2 * y_i * mu_hg[i])
            Re_hg = rho_hg[i] * U[i] * y[i]*2 / mu_hg[i]
            #
            # Bartz correlation
            Nu: float = (0.026
                            * Re_hg**0.8
                            * Pr_hg**0.6
                            * (Dt/R)**0.1
                            * (At/A[i])**0.9)

            sigma1 = (0.5 * (T_aw_hg / Tc) * (1+ (gamma_hg[i]-1) / 2 * M[i]**2) + 0.5) ** -0.68
            sigma2 = (1 + (gamma_hg[i]-1) / 2 * M[i]**2) ** -0.12
            h_hg_httrans = Nu * k_hg[i] / (2*y[i]) * sigma1 * sigma2

            # == RESISTANCE NETWORK == #
            R_w_hg = 1 / (2 * np.pi * y[i] * h_hg_httrans * ds)
            R_w = np.log((y[i] + t_wall) / y[i]) / (2 * np.pi * k_wall * ds)
            R_w_c = 1 / (h_c_httrans * Ph * ds * num_ch)

            # Compute all the heat fluxes to make sure they match
            R_tot = R_w_hg + R_w + R_w_c
            q_tot = (T_aw_hg - T_c_in) / R_tot

            # Hot gas side wall temp
            T_wg = T_aw_hg - q_tot * R_w_hg

            # Coolant side wall temp
            T_wc = T_wg - q_tot * R_w

            # Coolant temp with energy balance
            mdot_total = mdot
            T_c_out = T_c_in + q_tot / (mdot_total * cp_c)

            temp_resid = max(abs(T_wg - T_wall_hg), abs(T_wc - T_wall_c), abs(T_c_out - T_c_out_guess)) / T_aw_hg
            if temp_resid < tol:
                return T_c_0, T_wall_c, T_wall_hg, P_c, T_aw_hg, h_hg_httrans
            else:
                # No convergence yet
                T_c_out_guess = T_c_out
                T_wall_c = T_wc
                T_wall_hg = T_wg

        print(f"Slice {i} failed to converge!")
        return T_c_0, T_wall_c, T_wall_hg, P_c, T_aw_hg, h_hg_httrans

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
    Dt = np.min(y) * 2
    A = np.pi / 4 * y ** 2
    At = np.pi / 4 * Dt ** 2

    # == Channel Geometry == #
    depth: dict = data["C"]["depth_arr"]
    width: dict = data["C"]["width_arr"]
    num_ch: float = data["C"]["num_ch"]
    t_wall: float = data["W"]["thickness"]
    k_wall: float = data["W"]["k"]
    eps_dh: float = data["W"]["roughness"]

    # == dx Setup == #
    dx_seg = np.diff(x)
    dx_i_arr = np.empty(N, dtype=float)
    dx_i_arr[:-1] = dx_seg
    dx_i_arr[-1] = dx_seg[-1]

    """By this point the stagnation conditions are already established
    The Coolant channel and pressure drops have been calculated
    The following arrays should NOT be changing within this function"""

    # == Hot Gas Properties == #
    Tc: float = data["E"]["Tc"]
    P_hg: dict = data["Flow"]["P"]
    T_hg: dict = data["Flow"]["T"]
    M_hg: dict = data["Flow"]["M"]
    gamma_hg: dict = data["H"]["gamma"]
    mu_hg: dict = data["H"]["mu"]
    k_hg: dict = data["H"]["k"]
    cp_hg: dict = data["H"]["cp"]
    H_hg: dict = data["Flow"]["H"]
    mdot_hg: float = data["E"]["mdot"]
    rho_hg: dict = data["Flow"]["rho"]
    Pr_hg: dict = data["H"]["Pr"]

    # == Coolant Properties == #
    coolant: str = data["F"]["Type"]
    coolant: str = FLUID_MAP.get(coolant, coolant)
    mdot: float = data["F"]["mdot"]

    """These are initial conditions for the first slice ONLY
    All other slices will be set at the end of the previous slice
    These are the stagnation conditions however"""

    # == Hot Gas Properties == #
    H_hg_0: float = data["H"]["H"]
    C_hg: float = 0.023

    # == Coolant Properties == #
    T_c_0: float = data["F"]["T"]
    P_c_0: float = data["F"]["P"]
    rho_c_0: float = data["F"]["rho"]
    H_c_0: float = data["F"]["H"]
    C_c: float = 1
    g_c: float = 1

    # == Wall Properties == #
    T_wall_hg: float = data["W"]["InitialTemp"]
    T_wall_c: float = data["W"]["InitialTemp"]

    """These are all the storage items to keep track of data"""
    # == Stagnation Storage == #
    H_c_0_arr: np.ndarray = np.zeros(N, dtype=float)

    # == Static Storage == #
    P_c_arr: np.ndarray = np.zeros(N, dtype=float)
    H_c_arr: np.ndarray = np.zeros(N, dtype=float)
    rho_c_arr: np.ndarray = np.zeros(N, dtype=float)
    Re_c_arr: np.ndarray = np.zeros(N, dtype=float)
    v_c_arr: np.ndarray = np.zeros(N, dtype=float)
    h_hg_arr: np.ndarray = np.zeros(N, dtype=float)

    # == Temperature Storage == #
    T_wall_c_arr: np.ndarray = np.zeros(N, dtype=float)
    T_wall_hg_arr: np.ndarray = np.zeros(N, dtype=float)
    T_c_arr: np.ndarray = np.zeros(N, dtype=float)
    T_aw_arr: np.ndarray = np.zeros(N, dtype=float)

    # == Energy Storage == #
    q_c_arr: np.ndarray = np.zeros(N, dtype=float)

    # == Individual Values == #
    rho_c: float = np.nan

    st = AbstractState("HEOS", coolant)

    # Iterate through each slice
    for i in range(N - 1, -1, -1):

        progress = (N - 1 - i) / (N - 1) * 100
        print(f"\rCooling solver progress: {progress:5.1f}%", end="", flush=True)

        # Slice radius
        y_i = y[i]

        # Iterations take place within these functions
        if i == N - 1:
            T_c_0, T_wall_c, T_wall_hg, P_c, T_aw, h_hg = first_station_function(T_c_0=T_c_0, T_wall_hg=T_wall_hg,
                                                                           T_wall_c=T_wall_c, i=i)

        else:
            T_c_0, T_wall_c, T_wall_hg, P_c, T_aw, h_hg = other_station_function(T_c_0=T_c_0, T_wall_hg=T_wall_hg,
                                                                           T_wall_c=T_wall_c, P_c=P_c, i=i)

        # Commit slice info to a data array
        T_c_arr[i] = T_c_0
        T_wall_c_arr[i] = T_wall_c
        T_wall_hg_arr[i] = T_wall_hg
        T_aw_arr[i] = T_aw
        P_c_arr[i] = P_c
        h_hg_arr[i] = h_hg

    q = {"T_cool": T_c_arr,
         "T_wall_coolant": T_wall_c_arr,
         "T_wall_gas": T_wall_hg_arr,
         "h_hg": h_hg_arr,
         "P_c": P_c_arr}

    return q


def heat_transfer_solver_1(data: dict, max_iter=50, tol=1e-2):

    # ============ #
    # == SOLVER == #
    # ============ #
    energy_method = data["Solver"]["EnergyMethod"]

    # == Engine Geometry == #
    x = data["E"]["x"]
    y = data["E"]["y"]
    U = data["Flow"]["U"]
    N = len(x)

    # == Channel Geometry == #
    depth:  dict        = data["C"]["depth_arr"]
    width:  dict        = data["C"]["width_arr"]
    num_ch: float       = data["C"]["num_ch"]
    t_wall: float       = data["W"]["thickness"]
    k_wall: float       = data["W"]["k"]

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
    FLUID_MAP = {
        "LOX": "Oxygen",
        "GOX": "Oxygen",
        "RP-1": "n-Dodecane",
        "Kerosene": "n-Dodecane",
        "Kero": "n-Dodecane",
        "CH4": "Methane"
    }
    coolant = FLUID_MAP.get(coolant, coolant)
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
        if i != N-1:
            dx_i = np.sqrt((y[i] - y[i+1])**2 + (x[i] - x[i+1])**2)
        else:
            dx_i = 1e-6

        # Initial slice temps
        T_c_w_old = T_wall_c
        T_hg_w_old = T_wall_hg
        T_c_old = T_c_0

        # If first slice, static is equal to stagnation
        # Otherwise this will be set after convergence



        # Iterate multiple times to get the proper values
        for j in range(max_iter):

            # == COOLANT == #

            # If not the first slice, and then if first iteration or not
            if i != N-1:
                if j == 0:
                    rho_c: float = float(rho_c_arr[i+1])
                else:
                    # This is derived directly from the previous iteration, no manual adjusting of any kind
                    rho_c: float = CP.PropsSI("rhomass", "P", P_c, "H", H_c)

            else:
                if j == 0:
                    H_c: float = H_c_0
                    P_c: float = P_c_0
                    rho_c: float = rho_c_0

            # If first iteration, update stagnation enthalpy with eqn 9
            if j == 0:
                H_c_0: float = H_c_0_arr[i+1] + (q_c_arr[j+1] * dx_i / mdot)

            else:
                H_c_0: float = H_c_0_arr[i+1] + ((q_c_arr[j] + q_c_arr[j+1]) * dx_i / mdot)

            # Coolant velocity
            v_c: float = mdot / (rho_c * depth[i] * width[i] * num_ch)

            # Static enthalpy
            H_c: float = H_c_0 - (v_c**2 / 2)

            # Coolant Reynolds values (static and reference)
            mu_c: float = CP.PropsSI("viscosity", "P", P_c_arr[i + 1], "H", H_c_arr[i + 1])
            Re_c: float = mdot * y_i / (depth[i] * width[i] * num_ch * mu_c)

            Re_c_Ref: float = Re_c * (rho_hg[i] / rho_c) * (mu_c / mu_hg[i])

            # Average coolant reynolds number for better accuracy
            # Do not use if first slice
            if j != N-1:
                Re_c_avg: float = 0.5 * (Re_c_arr[i] + Re_c_arr[i+1])
                Re_c_Ref_avg: float = 0.5 * (Re_c_Ref[i] + Re_c_Ref_arr[i+1])
            else:
                Re_c_avg: float = Re_c
                Re_c_Ref_avg: float = Re_c_Ref

            # Friction factor using Moody diagram
            C1: float = C_c / 0.023
            if Re_c_Ref_avg < 2.2e3:
                f_c: float = 64 / Re_c_Ref_avg
            elif 2.2e3 <= Re_c_Ref_avg < 10e4:
                f_c: float = 4*C1*(0.0014+ (0.125/Re_c_Ref_avg**0.32))
            else:
                f_c: float = 0.778*C_c*Re_c_Ref_avg**-0.1021

            # Viscous Pressure drop (darcy)
            dP_c_f: float = f_c/(4*g_c) * ((rho_c + rho_c_arr[i+1])/(y_i + y[i+1])) * (v_c**2 + v_c_arr[i+1]**2) * dx_i

            # Momentum pressure drop
            dP_c_M: float = ((2 / ((N*depth[i+1]*width[i+1]) + (N*depth[i]*depth[i+1]))) *
                      mdot**2/g_c *
                      ((1 / (rho_c*depth[i]*width[i]*N)) - (1 / (rho_c_arr[i+1]*depth[i+1]*width[i+1]*N))))

            # New coolant pressure (eqn 15)
            P_c: float = P_c_arr[i+1] - (dP_c_f + dP_c_M)

            # Updated coolant wall properties
            st.update(CP.PT_INPUTS, P_c, T_wall_c)
            cp_w_c: float = st.cpmass()
            mu_w_c: float = st.viscosity()
            k_w_c: float = st.conductivity()
            P_c_w: float = st.p()
            H_c_w: float = st.hmass()

            # Coolant reference enthalpy (eqn 16)
            H_c_Ref: float = 0.5*(H_c + H_c_w) + 0.194*(H_c_0 - H_c)

            # Update coolant reference properties
            st.update(CP.PSmass_INPUTS, P_c, H_c_Ref)
            T_c_Ref: float = st.T()
            cp_c_Ref: float = st.cpmass()
            mu_c_Ref: float = st.viscosity()
            k_c_Ref: float = st.conductivity()
            P_c_Ref: float = st.p()
            rho_c_Ref: float = st.rhomass()

            # Update coolant static properties
            st.update(CP.PSmass_INPUTS, P_c, H_c)
            T_c: float = st.T()
            cp_c: float = st.cpmass()
            mu_c: float = st.viscosity()
            k_c: float = st.conductivity()
            P_c: float = st.p()
            rho_c: float = st.rhomass()

            # Coolant Prandtl Number reference
            Pr_c_Ref: float = cp_c_Ref * mu_c_Ref / k_c_Ref

            # Coolant adiabatic wall enthalpy
            H_c_aw: float = H_c + (Pr_c_Ref**(1/3) * (H_c_0 - H_c))

            # Coolant adiabatic wall temp
            T_c_aw: float = CP.PropsSI("T", "P", P_c, "H", H_c_aw)

            # Coolant heat transfer using dittus-boltzman
            d_hyd: float = 2 * (depth[i] * width[i]) / (depth[i] + width[i])
            Re_c_Ref: float = rho_c_Ref * v_c * d_hyd / mu_c_Ref

            if 0.1 <= Pr_c_Ref <= 1.0:
                Nu_c_Ref: float = 0.02155 * Re_c_Ref**0.8018 * Pr_c_Ref**0.7095
            elif 1.0 < Pr_c_Ref <= 3.0:
                Nu_c_Ref: float = 0.01253 * Re_c_Ref**0.8413 * Pr_c_Ref**0.6179
            else:
                Nu_c_Ref: float = 0.00881 * Re_c_Ref**0.8991 * Pr_c_Ref**0.3911

            h_c_httrans: float = Nu_c_Ref * k_c_Ref / d_hyd


            # == WALL == #
            # Update wall properties, updated from previous iteration
            Material_Properties(dic=data)
            k_wall = data["W"]["k"]
            # Wall heat transfer coefficient
            h_w_httrans: float = t_wall / k_wall


            # == HOT GAS == #
            H_wall_hg: float = H_hg[i] + cp_hg[i]*(T_wall_hg - T_hg[i])

            # Hot gas reference enthalpy
            H_hg_Ref: float= 0.5 * (H_wall_hg + H_hg[i]) + 0.18 * (H_hg_0 - H_hg[i])  # Place in loop

            # Hot gas reference temperature
            T_hg_Ref: float = CP.PropsSI("T", "P", P_hg, "H", H_hg_Ref)

            # Hot gas Prandtl reference number
            Pr_hg_Ref: float = cp_hg[i]*mu_hg[i]/k_hg[i]

            # Hot gas adiabatic wall temp
            H_hg_aw: float = H_hg[i] + (Pr_hg_Ref**(1/3) * (H_hg_0 - H_hg[i]))

            # Hot gas reynolds number reference
            Re_hg_Ref: float = 4 * mdot_hg * T_hg[i] / (np.pi * 2*y_i * mu_hg[i] * T_hg_Ref)

            # Hot gas side heat transfer coefficient
            h_hg_httrans: float = (C_hg * k_hg[i] / (y_i*2)) * Re_hg_Ref**0.8 * Pr_hg_Ref**0.3

            # == COMBINED HEAT FLUX == #

            # Basically a resistance setup
            h_overall: float = ((1/h_hg_httrans) + h_w_httrans + (1/h_c_httrans))**-1

            # Total heat flux q = H*(Taw - Tco)
            q_tot: float = h_overall * ((H_hg_aw/cp_hg[i]) - (H_c_Ref/cp_c_Ref))

            # New hot gas side wall temp
            T_hg_w_new: float = (H_hg_aw / cp_hg[i]) - (q_tot/h_hg_httrans)
            T_c_w_new: float = T_hg_w_new - (q_tot/h_w_httrans)
            T_c_new: float = T_c_w_new - (q_tot/h_c_httrans)


            # Now compare ALL the new temps with the previous iteraiton
            err_hg_w: float = abs(T_hg_w_new - T_hg_w_old) / T_hg_w_new
            err_c_w: float = abs(T_c_w_new - T_c_w_old) / T_c_w_new
            err_c: float = abs(T_c_new - T_c_old) / T_c_new
            if (err_hg_w or err_c or err_c_w) < tol:
                break

            # == UPDATE FOR NEXT ITERATION == #

            # Temps
            T_hg_w_old: float = T_hg_w_new
            T_c_w_old: float = T_c_w_new
            T_c_old: float = T_c_new

            # Coolant properties
            v_c_arr[i] = v_c
            P_c_arr[i] = P_c
            rho_c_arr[i] = rho_c
            Re_c_arr[i] = Re_c
            Re_c_Ref_arr[i] = Re_c_Ref

    q = {"V": v_c_arr}
    print(T_hg_w_new, T_c_w_new, T_c_new)