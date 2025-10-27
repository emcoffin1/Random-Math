from numpy.ma.core import concatenate

from HeatTransfer.bartz_formuals import bartz_heat_transfer, total_heat_flux
from MachSolver import mach_from_area_ratio as mach_eps
import numpy as np
from NozzleDesign import build_nozzle
import extra_utils as utils
import isentropic_flow as isen
import species, species_maker
import matplotlib.pyplot as plt



def nozzle_flow(eps, T0, P0, gamma, R):

    M = np.array([mach_eps(eps=e, gamma=gamma) for e in eps])
    # print(M)
    T = T0 / (1 + (gamma-1)/2 * M**2)
    P = P0 * (T/T0)**(gamma/(gamma-1))
    a = np.sqrt(gamma * R * T)
    U = M * a
    rho = P / (R * T)
    return {"M": M, "U": U, "T": T, "P": P, "rho": rho}

def norm_frac(d):
    s = sum(max(0.0, float(v)) for v in d.values())
    if s <= 0.0:
        return {k: 0.0 for k in d}
    return {k: max(0.0, float(v))/s for k, v in d.items()}

def compute_mach(T, R, gamma, v):
    a = (gamma * R * T) ** 0.5
    return v / max(a, 1e-9)

def main(Rt, T0, P0, Pa, gamma, R, cp, k, mu, gibbs, b_elem, species, mdot, frozen=True):
    # geometry
    x, y, a = build_nozzle(Pe=101300, Pc=P0, size=0.8, Rt=Rt, gamma=gamma, plots="no")

    # chamber equilibrium (upper bound) for h_t and initial frozen Y (if frozen=True)
    ch_st = isen.chamber_state(T0=T0, P0=P0, gibbs=gibbs, b_elem=b_elem, species=species)
    h_t = ch_st["h0"]
    Y_frozen = norm_frac(ch_st["Y"]) if frozen else None

    states = []
    n_total = len(a)
    i_throat = int(np.argmin(a))

    # sensible initial bracket and hint
    P_hi = P0 * 1.02
    P_lo = max(Pa * 0.5, 5e2)   # don’t go to near-zero kPa
    T_hint = T0

    # species order for plotting
    species_names = [sp.name for sp in gibbs.species]

    for i, A in enumerate(a):
        # Solve state at this area using last P/T as hints
        st = isen.solve_state(Ai=A, mdot=mdot, h_t=h_t, frozen=frozen, gibbs=gibbs,
                              b_elem=b_elem, species=species, Y_frozen=Y_frozen,
                              P_lo=P_lo, P_hi=P_hi, T_hint=T_hint)

        # keep the state
        states.append(st)

        # === Freeze logic: freeze when we hit/thru Mach 1 ===
        M = compute_mach(st["T"], st["R"], st["gamma"], st["v"]) if st["converged"] else 0.0
        # if not frozen and M >= 1.0:
        #     frozen = True
        #     Y_frozen = norm_frac(st["Y"])  # lock throat composition
        #     print(f"→ Freezing composition at x={x[i]:.3f} m (M≈{M:.2f})")
        if i > i_throat:
            frozen = True
        else:
            frozen = False



        # === Warm-starts for next step ===
        # Use last solved T as next hint
        if st["converged"] and np.isfinite(st["T"]):
            T_hint = st["T"]

        # Recenter pressure bracket around last solved P (tight bracket makes it fast & stable)
        if st["converged"] and np.isfinite(st["P"]):
            P_star = st["P"]
            P_lo = max(0.6 * P_star, Pa * 0.5, 5e2)
            P_hi = min(1.4 * P_star, P0 * 1.05)
            if P_lo >= P_hi:  # tiny numerical guard
                P_lo, P_hi = max(P_star * 0.8, 5e2), min(P_star * 1.2, P0 * 1.05)
        else:
            # fallback if we didn't converge
            P_lo = max(Pa * 0.5, 5e2)
            P_hi = P0 * 1.05

        # progress
        print(f"{(i+1)/n_total*100.0:.2f}%  --  Frozen: {frozen}    --  Mach: {M}   --  x: {x[i]}")

    # === Plot actual mass fractions (not delta) ===
    Y_mat = np.array([[norm_frac(st["Y"]).get(nm, 0.0) for nm in species_names] for st in states])
    plt.figure()
    for j, nm in enumerate(species_names):
        plt.plot(x, Y_mat[:, j], label=nm, linewidth=1.5)
    plt.xlabel("Length [m]"); plt.ylabel("Mass Fraction [-]"); plt.grid(True); plt.legend(ncol=3)
    plt.show()

    # Optional: velocity, T, P
    plt.figure(figsize=(9,5))
    plt.subplot(2,2,1); plt.plot(x,[st["v"] for st in states]); plt.grid(True); plt.ylabel("Velocity [m/s]")
    plt.subplot(2,2,2); plt.plot(x,[st["T"] for st in states]); plt.grid(True); plt.ylabel("Temperature [K]")
    plt.subplot(2,2,3); plt.plot(x,[st["P"] for st in states]); plt.grid(True); plt.ylabel("Pressure [Pa]"); plt.xlabel("Length [m]")
    plt.subplot(2,2,4); plt.plot(x, a)
    plt.tight_layout(); plt.show()


    # l = [v_vals, T_vals, P_vals]
    # labels = ["Velocity [m/s]", "Temperature [K]", "Pressure [Pa]"]
    # # labels = ["Mach Number", "Velocity [m/s]", "Temperature [K]", "Pressure [Pa]", "Density [kg/m³]"]
    # utils.plot_flow_char(x=x, data=l, labels=labels)
    # # utils.plot_flow_field(x, y, T, "Temp", mode=2)
    # utils.convert_to_func(x,y)


def main_basic(Pe=101325, Pc=2.013e6, Tc=3200, size=0.8, gamma=1.22, Rt=0.05, R=350, mu=8.617e-4, k=0.5937):
    # Build nozzle
    x, y, a = build_nozzle(Pe=Pe, Pc=Pc, size=size, gamma=gamma, Rt=Rt, plots="no")

    # Convert to ratio
    a_min = min(a)
    eps = a/a_min

    # Isolate subsonic and supersonic with a negative sign (will be zeroed out eventually)
    ind = np.where(eps == 1.0)[0][0]
    eps[:ind] *= -1

    # Solve isentropic relations
    flow = nozzle_flow(eps=eps, T0=Tc, P0=Pc, gamma=gamma, R=R)
    flows = [flow["M"], flow["U"], flow["T"], flow["P"], flow["rho"]]
    names = ["M", "U", "T", "P", "rho"]
    subnames = [None, None, None, None, None]
    # utils.plot_flow_char(x=x, data=flows, labels=names)

    # Solve for heat transfer
    cp = gamma * R / (gamma - 1)
    q = bartz_heat_transfer(gamma=gamma, R=R, Pc=Pc, rt=Rt, x=x, y=y, cp=cp, k=k, mu=mu,
                            T=flow["T"], Tc=Tc, M=flow["M"])
    flows1 = [q["qdot"], (q["T_wi"], q["T_wo"])]
    names1 = ["Heat Flux q", "Temps"]
    subnames1 = [None, ["Inner Temps", "Outer Temps"]]

    # Tc_out is the temperature of coolant leaving the regens and entering the injector
    # Compute the total heat transfer and mdot
    Tc_out = 350
    Q = total_heat_flux(qdot=q["qdot"], x=x, y=y, cp=cp, Tc_in=q["T_ci"], Tc_out=Tc_out)
    print(f"Mass flow rate (first pass): {Q["mdot"]:.3f}kg/s")
    print(f"Total heat flux (Q): {Q["Qtotal"]:.2f}W")
    flows2 = [Q["Q"]]
    names2 = ["Total Q"]
    subnames2 = [None]


    flows = flows + flows1 + flows2
    names = names + names1 + names2
    subnames = subnames + subnames1 + subnames2
    # utils.plot_flow_chart(x=x, data=flows, labels=names, sublabels=subnames)

    utils.plot_flow_field(x, y, data=q["qdot"], label="Heat Flux")


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
    mu = 8.617e-4  # Pa·s

    # Material Props (SI)


    # # Generate Species and Gibbs solver
    # spec, elem = species.get_species_data()
    # b_elem, total = species_maker.get_b_elem(fuel_formula="C12H26", oxidizer_formula="O2", OF_ratio=OF)
    # gibbs = species.GibbsMinimizer(species=list(spec.values()), elements=elem)
    #
    # main(Rt=Ri, T0=T0, P0=P0, Pa=101325, gamma=gamma, R=Rgas, k=k, mu=mu, cp=cp, gibbs=gibbs,
    #      b_elem=b_elem, species=spec, mdot=mdot, frozen=True)
    main_basic()