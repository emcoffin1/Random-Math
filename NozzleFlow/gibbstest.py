from species import *
import numpy as np

R = 8.314462618  # J/mol-K
PREF = 101325.0

def project_to_elements(A, b_vec, n, eps=1e-12):
    """
    Exact element closure: n <- n + A^T (AA^T)^-1 (b - A n)
    """
    Ne = A.shape[0]
    ATA = A @ A.T
    lam_corr = np.linalg.solve(ATA + eps*np.eye(Ne), b_vec - A @ n)
    return np.maximum(n + A.T @ lam_corr, 1e-50)

def total_G_over_RT(species_list, n, T, P):
    n_tot = np.sum(n)
    y = np.clip(n / n_tot, 1e-300, 1.0)
    g0 = np.array([sp.g_mol(T) for sp in species_list])  # J/mol
    return np.sum(n * (g0/(R*T) + np.log(y*P/PREF)))


class StableGibbsMinimizer:
    """
    Robust version:
      - Safe log/exp update
      - Element projection each iter
      - Damping
    """
    def __init__(self, species, elements, damping=0.4):
        self.species = species
        self.elements = elements
        self.A = np.array([sp.elem_vec for sp in species]).T  # (Ne x Ns)
        self.damping = damping

    def solve(self, T, P, b_elem, tol=1e-8, max_iter=300, return_diag=False):
        Ns = len(self.species)
        Ne = len(self.elements)
        A = self.A
        b_vec = np.array([b_elem[e] for e in self.elements], dtype=float)

        # Initial guess: even split scaled to roughly match elements
        n = np.ones(Ns, dtype=float) / Ns
        n = project_to_elements(A, b_vec, n)

        G_hist = []
        for it in range(max_iter):
            n_tot = np.sum(n)
            y = np.clip(n / n_tot, 1e-300, 1.0)

            # Chemical potentials
            g0 = np.array([sp.g_mol(T) for sp in self.species])  # J/mol
            mu = g0 + R*T*np.log(y*P/PREF)

            # Element residual
            res = b_vec - A @ n
            if np.linalg.norm(res) < tol:
                break

            # Solve for Lagrange multipliers for the residual step
            # (this is only used to shape the update direction)
            lam = np.linalg.solve(A @ A.T + 1e-12*np.eye(Ne), res)

            # Safe log/exp update
            # NOTE the sign: -(mu - A^T lam)/(RT) worked best in practice
            logn_prop = -(mu - A.T @ lam) / (R*T)

            # normalize & clip before exp to avoid overflow/underflow
            logn_prop -= np.max(logn_prop)
            logn_prop = np.clip(logn_prop, -50.0, 50.0)
            n_prop = np.exp(logn_prop)

            # Project exactly to element totals
            n_prop = project_to_elements(A, b_vec, n_prop)

            # Damped mix
            n_new = np.maximum(self.damping*n_prop + (1.0-self.damping)*n, 1e-50)

            # Optional: Armijo-like safeguard (accept only if G decreases)
            G_now = total_G_over_RT(self.species, n, T, P)
            G_try = total_G_over_RT(self.species, n_new, T, P)
            if G_try > G_now:   # if worse, soften the step once
                n_new = np.maximum(0.5*n_prop + 0.5*n, 1e-50)
                G_try = total_G_over_RT(self.species, n_new, T, P)

            n = n_new
            if return_diag:
                G_hist.append(G_try)

        out = {sp.name: ni for sp, ni in zip(self.species, n)}
        if return_diag:
            return out, {"iters": it+1, "G_hist": G_hist, "resid": float(np.linalg.norm(b_vec - A @ n))}
        return out

def report_fractions(y_eq, species_dict):
    import numpy as np
    names = list(y_eq.keys())
    n = np.array([y_eq[k] for k in names])
    nt = n.sum()
    y = n/nt  # mole fractions
    # mass fractions (if you want them for mix properties)
    mw = np.array([species_dict[k].mw for k in names])
    mass = n*mw
    w = mass/mass.sum()

    print("\n-- Mole fractions (sum=%.6f) --" % y.sum())
    for k, yi in sorted(zip(names, y), key=lambda kv: kv[1], reverse=True)[:10]:
        print(f"{k:5s}: {yi:.4f}")

    print("\n-- Mass fractions (sum=%.6f) --" % w.sum())
    for k, wi in sorted(zip(names, w), key=lambda kv: kv[1], reverse=True)[:10]:
        print(f"{k:5s}: {wi:.4f}")



def quick_trends(solver, b_elem, spec_all, elements, Ts=(2400,3000,3600), Ps=(1e5,5e6,1e7)):
    import numpy as np
    species_list = list(spec_all.values())
    names = [s.name for s in species_list]
    def molefrac(y_eq):
        n = np.array([y_eq.get(k,0.0) for k in names]); y = n/n.sum()
        return {k: yi for k, yi in zip(names, y)}

    print("\n== T sweep at P=5 MPa ==")
    for T in Ts:
        y_eq, _ = solver.solve(T=T, P=5e6, b_elem=b_elem, return_diag=True)
        y = molefrac(y_eq)
        print(f"T={T}K | H2O {y.get('H2O',0):.3f} CO2 {y.get('CO2',0):.3f} CO {y.get('CO',0):.3f} H {y.get('H',0):.3f} OH {y.get('OH',0):.3f}")

    print("\n== P sweep at T=3000 K ==")
    for P in Ps:
        y_eq, _ = solver.solve(T=3000, P=P, b_elem=b_elem, return_diag=True)
        y = molefrac(y_eq)
        print(f"P={P:.2e} | H2O {y.get('H2O',0):.3f} CO2 {y.get('CO2',0):.3f} CO {y.get('CO',0):.3f} H {y.get('H',0):.3f} OH {y.get('OH',0):.3f}")


def probe_thermo(spec, T=3000.0):
    # 1) Check Tmid values (should be ~1000 K for most)
    print("== Tmid values ==")
    for nm, sp in spec.items():
        print(f"{nm:4s} Tmid={sp.temp_cutoff}")

    # 2) Check coeff counts (NASA9: 7 poly coeffs + 2 integration constants per range)
    print("\n== coeff lengths ==")
    for nm, sp in spec.items():
        lo, hi = sp.coeffs["low"], sp.coeffs["high"]
        ilo, ihi = sp.int_const["low"], sp.int_const["high"]
        print(f"{nm:4s} low:{len(lo)} hi:{len(hi)}  int_low:{len(ilo)} int_high:{len(ihi)}")

    # 3) Evaluate cp/R, h/RT, s/R at T and around Tmid for continuity
    def dimless(sp, T):
        R = 8.314462618
        return (sp.cp_mol(T)/R, sp.h_mol(T)/(R*T), sp.s_mol(T)/R)
    print("\n== values at T=3000 K (cp/R, h/RT, s/R) ==")
    for nm in ["H2","O2","H2O","CO","CO2","OH","H","O"]:
        if nm in spec:
            sp = spec[nm]
            cpr, hrt, sr = dimless(sp, T)
            print(f"{nm:4s} cp/R={cpr:8.3f}  h/RT={hrt:8.3f}  s/R={sr:8.3f}")


# === BASIC TEST ===
if __name__ == "__main__":
    spec_all, elements_all = get_species_data()
    # species_list = list(spec_all.values())
    # elements_list = ['C', 'H', 'O', 'N']  # must match how you built elem_vec
    #
    # solver = StableGibbsMinimizer(species_list, elements_list, damping=0.4)
    #
    # # Example: decane/oxygen at OF=2.6 (your helper)
    # b_elem, _ = get_b_elem("C12H26", "O2", OF_ratio=2.6)
    #
    # y_eq, diag = solver.solve(T=3000.0, P=5e6, b_elem=b_elem, return_diag=True)
    # print("== Full-set test ==")
    # top = sorted(y_eq.items(), key=lambda kv: kv[1], reverse=True)[:8]
    # for k, v in top:
    #     print(f"{k:5s}: {v:.6e}")
    # print(f"iters={diag['iters']}, elem resid={diag['resid']:.2e}")
    #
    # # call after the solve in your Test B:
    # report_fractions(y_eq, spec_all)
    probe_thermo(spec=spec_all, T=3000.0)
