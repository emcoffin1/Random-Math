from species import *
import numpy as np



def test_gibbs_minimizer():
    print("=== Testing Gibbs Minimizer ===")
    spec_dict, elem_list = get_species_data()

    # IMPORTANT: Build species/elements in the SAME order the solver uses
    species_list = list(spec_dict.values())              # keep this list for names, mw, etc.
    gibbs = GibbsMinimizer(species_list, elem_list)      # solver stores this order internally

    # === Simple stoichiometric-ish combustion mixture ===
    b_elem, _ = get_b_elem("H2", "O2", OF_ratio=8.0)     # element totals (moles of atoms)
    T = 3000.0
    P = 5e6

    try:
        y_eq = gibbs.solve(T=T, P=P, b_elem=b_elem)
    except Exception as e:
        print("❌ FAILED: Exception during solve:", e)
        return False

    # Align n with solver's species order (NOT dict order)
    names = [sp.name for sp in gibbs.species]
    n = np.array([y_eq[name] for name in names], dtype=float)

    # Use the solver's own A and elements ordering
    A = gibbs.A
    b_vec = np.array([b_elem[e] for e in gibbs.elements], dtype=float)

    # === Check 1: Nonnegative species ===
    if np.any(n < 0):
        print("❌ FAILED: Negative mole numbers found")
        return False

    # === Check 2: Element balance ===
    resid_vec = b_vec - A @ n
    resid = np.linalg.norm(resid_vec)
    if resid > 1e-6 * max(1.0, np.linalg.norm(b_vec)):
        # Print per-element residual to help debugging if needed
        per_elem = {e: resid_vec[i] for i, e in enumerate(gibbs.elements)}
        print(f"❌ FAILED: Element balance off (‖resid‖={resid:.2e}) | per-element: {per_elem}")
        return False

    # === Check 3: Dominant species sanity ===
    nt = n.sum()
    y = n / nt
    top = sorted(zip(names, y), key=lambda kv: kv[1], reverse=True)[:5]
    major = [k for k, v in top if v > 1e-6]
    if not any(s in major for s in ("H2O", "OH", "H2", "O2")):
        print("❌ FAILED: No reasonable combustion products in top species.")
        print("Top species:", major)
        return False

    print(f"✅ PASS: Converged at T={T:.0f} K, P={P/1e6:.1f} MPa")
    print(f"   Element balance residual = {resid:.2e}")
    print("   Top mole fractions:")
    for k, v in top:
        print(f"     {k:5s}: {v:.6f}")
    return True


if __name__ == "__main__":
    ok = test_gibbs_minimizer()
    if not ok:
        print("Gibbs minimizer failed basic validation.")
