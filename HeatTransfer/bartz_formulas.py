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
    """
    Equations used:
    Coolant:
        h_out - h_in = sum(Qdot / mdot)
        T_out - T_in = sum(Qdot / mdot / cp)

    Wall - Coolant:
        T_wall - T_cool_out = Qdot * R_wall_coolant

    Wall - Hot gas:
        T_aw - T_wall = Qdot * R_bartz

    """
    """Engine and Hot Gas Information"""
    # =========================== #
    # ==  Dictionary breakdown == #
    # =========================== #
    """Random Solver Controls"""
    film_cooling                = info["Solver"]["FilmCool"]
    energy_method               = info["Solver"]["EnergyMethod"]

    HotGas_Properties(dic=info)
    FLUID_MAP = {
        "LOX": "Oxygen",
        "GOX": "Oxygen",
        "RP-1": "n-Dodecane",
        "Kerosene": "n-Dodecane",
        "Kero": "n-Dodecane",
        "CH4": "Methane"
    }

    coolant: str = info["F"]["Type"]
    coolant: str = FLUID_MAP.get(coolant, coolant)

    """Engine and Hot gas information"""
    Tc                          = float(info["E"]["Tc"])
    T_hg                            = info["Flow"]["T"]
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
    eps:        float           = info["W"]["roughness"]

    """Geometry and flow arrays"""
    x:          list            = info["E"]["x"]
    y:          list            = info["E"]["y"]
    M:          list            = info["Flow"]["M"]
    N:          float           = len(x)  # Number of x points
    depth:      list            = info["C"]["depth_arr"]
    width:      list            = info["C"]["width_arr"]
    ch_num:     float           = info["C"]["num_ch"]

    """Storage"""
    Taw:        np.ndarray      = np.zeros(N, dtype=float)
    h_hg:       np.ndarray      = np.zeros(N, dtype=float)
    h_wc:       np.ndarray      = np.zeros(N, dtype=float)
    T_wall_coolant: np.ndarray  = np.zeros(N, dtype=float)
    T_wall_gas: np.ndarray      = np.zeros(N, dtype=float)
    T_c_out:    np.ndarray      = np.zeros(N, dtype=float)
    Q_dot:      np.ndarray      = np.zeros(N, dtype=float)
    R_hg_w_arr: np.ndarray      = np.zeros(N, dtype=float)
    R_w_c_arr:  np.ndarray      = np.zeros(N, dtype=float)
    R_w_w_arr:  np.ndarray      = np.zeros(N, dtype=float)
    Re_arr:     np.ndarray      = np.zeros(N, dtype=float)
    v_arr:      np.ndarray      = np.zeros(N, dtype=float)
    Dh_arr:     np.ndarray      = np.zeros(N, dtype=float)
    P_bulk_arr: np.ndarray      = np.zeros(N, dtype=float)
    dp_arr: np.ndarray = np.zeros(N, dtype=float)
    qpp_hg: np.ndarray = np.zeros(N, dtype=float)
    info["F"]["rho_arr"] : np.ndarray()       = np.zeros(N, dtype=float)
    info["F"]["mu_arr"] : np.ndarray()       = np.zeros(N, dtype=float)

    h_in_arr = np.zeros(N)
    h_out_arr = np.zeros(N)

    """
    Initial Conditions
    Initial temperature is set as   
    """
    info["C"]["h"]              = np.zeros_like(x)
    info["C"]["Nu"]             = np.zeros_like(x)
    info["C"]["Re"]             = np.zeros_like(x)
    info["C"]["T"]              = np.zeros_like(x)

    converge_count = 0

    st = AbstractState("HEOS", coolant)
    # st = info["F"]["State"]

    # == INITIAL CONDITIONS == #
    T_coolant_in = T_f
    P_coolant_in = info["F"]["StartingPressure"]
    st.update(CP.PT_INPUTS, P_coolant_in, T_coolant_in)
    h_coolant_in = st.hmass()

    """Init conditions for energy conservation check"""
    T_in = T_coolant_in
    P_in = P_coolant_in
    h_in = h_coolant_in

    """March along x"""
    # March through all slices, i
    # Starts from -1 and works backwards (-1, -2, -3, ...)
    # This is done as fuel (coolant) enters the engine from the nozzle exit side
    for i in range(N-1, -1, -1):

        progress        = (N - 1 - i) / (N - 1) * 100
        print(f"\rCooling solver progress: {progress:5.1f}%", end="", flush=True)

        # Slice Radius
        y_i             = float(y[i])

        # Slice length
        if i != N-1:
            ds = np.sqrt((y[i] - y[i + 1]) ** 2 + (x[i] - x[i + 1]) ** 2)
        else:
            ds = np.sqrt((y[i-1] - y[i]) ** 2 + (x[i-1] - x[i]) ** 2)

        # Coolant geometry
        Dh_c = 2 * depth[i] * width[i] / (depth[i] + width[i])
        eps_dh = eps / Dh_c

        # Last computed values for use after convergence
        Q_i             = 0.0
        R_w_c           = np.nan
        R_w_w           = np.nan
        R_hg_w          = np.nan
        h_coolant       = np.nan
        Nu = Re = Dh = Ah = V = n_ef = np.nan
        T_coolant_out   = np.nan
        h_coolant_out   = np.nan
        P_coolant_out = np.nan
        T_bulk = np.nan
        P_bulk = np.nan

        # ==================================== #
        # == COOLANT TEMPERATURE ITERATIONS == #
        # ==================================== #

        # Iterate a max of j times
        for j in range(max_iteration):

            # == PRESSURE DROP == #
            # This needs to be iterated because the temperature influences the props
            if j == 0:
                # First iteration exception
                # This must be done since we don't actually know
                # the outlet temp yet
                T_coolant_out = T_coolant_in

            # Reynolds number
            st.update(CP.PT_INPUTS, P_coolant_in, T_coolant_in)
            rho_c = st.rhomass()
            v_c = (mdot_f/ch_num) / (rho_c * depth[i] * width[i])
            Re_c = rho_c * v_c * Dh_c / st.viscosity()

            # Laminar
            if Re_c < 2300:
                print("Lainar")
                f = 64 / Re_c
            # Transient
            elif 2300 < Re_c < 4000:
                print("Transient")
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

            dP_c = f * (ds / Dh_c) * (rho_c * v_c ** 2 / 2)

            # Now update the pressure at this slice
            # This is NOT an energy method
            # And this will not change
            P_coolant_out = P_coolant_in - dP_c

            # Compute bulk values, accounts for variations
            P_bulk = 0.5 * (P_coolant_in + P_coolant_out)
            T_bulk = 0.5 * (T_coolant_in + T_coolant_out)

            # Coolant heat transfer
            h_coolant, Nu, Re, Dh, A_eff, V = dittus_appro(dx=ds, dic=info, width=width[i], depth=depth[i], step=i,
                                                           T=T_bulk, P=P_bulk, st=st, f=f)

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
            R_hg_w              = 1 / (2 * np.pi * y_i * ds * h_hg_i)

            # Wall conduction
            R_w_w               = np.log((y_i+t_w) / y_i) / (2 * np.pi * ds * k_w)

            # Wall to coolant
            R_w_c               = 1 / (h_coolant * A_eff)

            # Total resistance network
            R_total             = R_hg_w + R_w_w + R_w_c

            # Using the initial heat flux guess, determine the expected wall temp
            # T_wall_predictor = T_bulk + Q_i * R_w_c
            Q_new = (Taw_i - T_bulk) / R_total

            if energy_method:
                # Energy method to get coolant out temp
                h_coolant_out = h_coolant_in + Q_new / mdot_f
                # Convert that to temp
                st.update(CP.HmassP_INPUTS, h_coolant_out, P_bulk)

                # Update the coolant value for the next iteration
                T_coolant_out = st.T()

                if P_bulk < st.p_critical():
                    st.update(CP.PQ_INPUTS, P_bulk, 0)
                    if T_coolant_out > st.T():
                        print(f"Coolant is above saturation temperature at slice: {i} -- Temp: {T_coolant_out} -- Pres: {P_bulk}")
                cp_bulk = st.cpmass()
                rho_bulk = st.rhomass()
                mu_bulk = st.viscosity()

            else:
                # Temp method to get coolant out temp
                st.update(CP.PT_INPUTS, P_bulk, T_bulk)
                cp_bulk = st.cpmass()
                T_coolant_out    = T_coolant_in + Q_new / (mdot_f * cp_bulk)

            T_bulk = 0.5 * (T_coolant_in + T_coolant_out)

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
        dp_arr[i]   = dP_c

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
        T_wall_coolant[i] = T_bulk + Q_i * R_w_c
        T_wall_gas[i] = Taw[i] - Q_i * R_hg_w
        qpp_hg[i] = h_hg[i] * (Taw[i]-T_wall_gas[i])


        # Energy Cons storage
        h_in_arr[i] = h_coolant_in
        h_out_arr[i] = h_coolant_out


        # Advance coolant enthalpy for next slice
        if energy_method:
            # Update these for channel convergence
            info["F"]["H"] = float(h_coolant_out)
            info["F"]["rho_arr"][i] = rho_bulk
            info["F"]["mu_arr"][i] = mu_bulk

        else:
            info["F"]["T"] = T_coolant_out
        T_coolant_in = T_coolant_out
        P_coolant_in = P_coolant_out
        h_coolant_in = h_coolant_out

    # == ENERGY CONSERVATION == #
    P_out = P_bulk_arr[0]
    T_out = T_c_out[0]
    if energy_method:
        h_out = h_coolant_out
    else:
        st.update(CP.PT_INPUTS, P_out, T_out)
        h_out = st.hmass()

    Q_total = np.sum(Q_dot)
    Q_coolant = mdot_f * (h_out - h_in)
    den = max(abs(Q_total), abs(Q_coolant), 1e-9)
    err_frac = (Q_total - Q_coolant) / den
    err_pct = 100 * err_frac

    Q_slice_coolant = mdot_f * (h_out_arr - h_in_arr)
    slice_error = Q_dot - Q_slice_coolant

    # print("\n", np.sum(dp_arr)*0.000145038)

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
           "P_c": P_bulk_arr,
           "dP_arr": np.sum(dp_arr),
           "qpp_hg": qpp_hg,}


    if err_pct > 1.0:
        print(f"\nCooling solution complete with {err_pct:.2e}% conservation.", end="", flush=True)
        print(f"\nSlice RMS energy error: {np.sqrt(np.mean(slice_error**2)):.2e} W", end="", flush=True)
        print(f"\nSlice MAX energy error: {np.max(np.abs(slice_error)):.2e} W", end="", flush=True)
    print(f"\nCooling solution complete.", end="", flush=True)
    print()
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



def dittus_appro(dx:float, dic:dict, depth: int, width: int, step: int, P:float, T:float, st:object, f:float):
    """
    Computes convective heat transfer coefficient for the coolant in channels
    :param dic: dictionary of all information
    :param dimension: Complexity of solver, 0 for simple solves all points at once
    :param step: position in engine
    """
    i = step

    mdot =  dic["F"]["mdot"]

    st.update(CP.PT_INPUTS, P, T)
    rho = st.rhomass()
    mu = st.viscosity()
    k = st.conductivity()
    cp = st.cpmass()

    y = dic["E"]["y"][i]
    # Run the max number of channels if not already computed
    if dic["C"]["num_ch"] is not None:
        num_ch = dic["C"]["num_ch"]
        type, thickness_w, spacing, height = (dic["C"]["Type"].lower(), dic["W"]["thickness"],
                                                     dic["C"]["spacing"], dic["C"]["height"])

    else:
        raise ValueError(f"Number of channels is not defined")

    # Compute hydraulic diameter depending on shape type
    if type == "square":
        chan_depth = depth
        chan_width = width
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
        n_ef = min(1.0, num_ch * Ph/(2 * np.pi * y))

        # Velocity from mass flow rate equation
        V = mdot_ch / (rho * A)

        # Reynolds number using hydaulic diameter
        Re = rho * V * Dh / mu

        # Prandtl Number
        Pr = cp * mu / k

        # Nusselt
        # Dittus
        # Nu = 0.023 * Re**0.8 * Pr**0.4

        # Gnielinski
        Nu = ((f/8) * (Re-1000)*Pr) / (1 + 12.7*(f/8)**0.5*(Pr**(2/3)-1)) # * (1 + (Dh/dx)**(2/3))

        # Heat Transfer coefficient
        h_f = Nu * k / Dh

        # Fin efficiency
        m = np.sqrt(2*h_f/(k*spacing))
        depth_c = depth + 0.5*spacing
        n_ef = np.tanh(m*depth_c)/(m*depth_c)

        A_base = width*dx
        A_fin = 2*depth*dx
        A_eff = num_ch * (A_base + n_ef*A_fin)

        if np.isnan(h_f):
            raise ValueError(f"h_f is not defined\n"
                             f"Nu = {Nu}\n"
                             f"k = {k}\n"
                             f"Dh = {Dh}\n"
                             f"Re = {Re}\n"
                             f"cp = {cp}\n"
                             f"mu = {mu}\n"
                             f"Pr = {Pr}\n")


        return h_f, Nu, Re, Dh, A_eff, V

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
    mu_arr      = data["F"]["mu_arr"]
    if len(mu_arr) != 0:
        mu = mu_arr

    depth       = data["C"]["depth_arr"]
    width       = data["C"]["width_arr"]
    eps         = data["W"]["roughness"]

    N           = len(data["E"]["x"])
    y           = data["E"]["y"]
    x           = data["E"]["x"]
    dx          = np.zeros(N)

    for i in range(N - 1, -1, -1):
        # Slice length
        if i != N - 1:
            dx[i] = np.sqrt((y[i] - y[i + 1]) ** 2 + (x[i] - x[i + 1]) ** 2)
        else:
            dx[i] = np.sqrt((y[i - 1] - y[i]) ** 2 + (x[i - 1] - x[i]) ** 2)

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
    data["F"]["P"] = data["Injector"]["dP"] + sum_array

    # Saves the inlet pressure of the coolant
    data["F"]["StartingPressure"] = data["F"]["P"]

    if data["Solver"]["EnergyMethod"]:
        data["Solver"]["EnergyMethod"] = False
        Fluid_Properties(dic=data)
        data["Solver"]["EnergyMethod"] = True


