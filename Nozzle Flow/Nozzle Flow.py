from MachSolver import mach_from_area_ratio_supersonic as mach_eps
import numpy as np
from NozzleDesign import build_nozzle
import extra_utils as utils
import isentropic_flow as isen
import species, species_maker
import matplotlib.pyplot as plt


def should_equilibrate(T, rho, u, species, Y, Lc=0.05, Da_crit=5.0):
    """
    Determines if flow is in equilibrium or frozen,
    based on temperature, pressure, density, velocity, and species.
    :returns bool: True if frozen, False if equilibrium
    """
    R_univ = 8.314462618  # J/mol-K
    # Convert dict Y → array in species order
    Y = np.array([Y.get(sp.name, 0.0) for sp in species])

    # === Mixture molecular weight === #
    mws = np.array([sp.mw for sp in species])
    inv_MW_mix = np.sum(Y / mws)
    MW_mix = 1.0 / inv_MW_mix

    # === Flow timescale === #
    flow = Lc / max(u, 1e-30)

    # === Chemical timescale === #
    A = 2.0e14
    n = 0.0
    Ea = 7.1e4 * R_univ
    k_eff = A * T**n * np.exp(-Ea / (R_univ * T))

    conc = rho / MW_mix

    chem = 1 / max(k_eff*conc, 1e-30)

    # === Damkohler number === #
    Da = flow / chem

    return Da < Da_crit


def nozzle_flow(eps, T0, P0, gamma, R):

    M = np.array([mach_eps(eps=e, gamma=gamma) for e in eps])
    T = T0 / (1 + (gamma-1)/2 * M**2)
    P = P0 * (T/T0)**(gamma/(gamma-1))
    a = np.sqrt(gamma * R * T)
    U = M * a
    rho = P / (R * T)
    return {"M": M, "U": U, "T": T, "P": P, "rho": rho}


def main(Rt, T0, P0, Pa, gamma, R, cp, k, mu, gibbs, b_elem, species, mdot, frozen=True):
    # Get nozzle geometry
    x, y, a = build_nozzle(Pe=101300, Pc=P0, size=0.8, Rt=Rt, gamma=gamma, plots="no")

    # Initial chamber state
    ch_st = isen.chamber_state(T0=T0, P0=P0, gibbs=gibbs, b_elem=b_elem, species=species)
    h_t = ch_st["h0"]
    Y_frozen = ch_st["Y"] if frozen else None

    # Start saving and hints
    states =[]
    P_hi = P0
    P_lo = max(Pa*0.05, 500.0)
    T_hint = T0

    n_total = len(a)
    for i, A in enumerate(a, start=1):

        # Compute next step at next area (A)
        st = isen.solve_state(Ai=a[i-1], mdot=mdot, h_t=h_t, frozen=frozen, gibbs=gibbs, b_elem=b_elem,
                              species=species, Y_frozen=Y_frozen, P_lo=P_lo, P_hi=P_hi, T_hint=T_hint)

        # Check if previous flow was frozen or not
        frozen_new = should_equilibrate(T=st["T"], rho=st["rho"], u=st["v"], species=list(species.values()), Y=st["Y"])
        # if frozen_new != frozen:
        #     print(f"Switching to {'frozen' if frozen_new else 'equilibrium'} at x={x[i-1]:.3f} m")
        #     frozen = frozen_new
        if i > 0.50*len(x) :
            frozen = False
        else:
            frozen = True

        # Update hints and bounds
        states.append(st)

        P_hi = max(P_lo, 1e3)
        P_lo = min(P_hi, 2*P0)

        # Update progress
        print(f"{i/n_total*100.0:.2f}%")

    y_name = list(states[0]["Y"].keys())
    # y_val = np.array([[s["Y"][sp] for sp in y_name] for s in states])
    y_val = np.array([[st["Y"].get(n, 0.0) for n in y_name] for st in states])
    dy = y_val - y_val[0:1]


    plt.figure()
    for i, sp in  enumerate(y_name):
        plt.plot(x, dy[:, i], label=sp)
    plt.xlabel("Length [m]")
    plt.ylabel("Mass Fraction [-]")
    plt.legend()
    plt.grid(True)
    plt.show()


    # Plot flow nozzle values
    v_vals = [s["v"] if s["converged"] else 0.0 for s in states]
    T_vals = [s["T"] if s["converged"] else 0.0 for s in states]
    P_vals = [s["P"] if s["converged"] else 0.0 for s in states]

    l = [v_vals, T_vals, P_vals]
    labels = ["Velocity [m/s]", "Temperature [K]", "Pressure [Pa]"]
    # labels = ["Mach Number", "Velocity [m/s]", "Temperature [K]", "Pressure [Pa]", "Density [kg/m³]"]
    utils.plot_flow_char(x=x, data=l, labels=labels)
    # utils.plot_flow_field(x, y, T, "Temp", mode=2)
    # utils.convert_to_func(x,y)


if __name__ == '__main__':
    Ri = 0.05  # m (throat radius)

    T0 = 3493   # K
    P0 = 2.013e6  # Pa
    OF = 2.6
    mdot = 1.78
    gamma = 1.23
    Rgas = 421.6  # J/kg-K


    # Properties (SI)
    cp = 2852.7  # J/kg-K  (2.8527 kJ/kg-K)
    k = 0.5937  # W/m-K
    mu = 8.617e-4  # Pa·s  (if you meant 0.8617 mPa·s)

    # Generate Species and Gibbs solver
    spec, elem = species.get_species_data()
    b_elem, total = species_maker.get_b_elem(fuel_formula="C12H26", oxidizer_formula="O2", OF_ratio=OF)
    gibbs = species.GibbsMinimizer(species=list(spec.values()), elements=elem)

    main(Rt=Ri, T0=T0, P0=P0, Pa=101325, gamma=gamma, R=Rgas, k=k, mu=mu, cp=cp, gibbs=gibbs,
         b_elem=b_elem, species=spec, mdot=mdot, frozen=True)
