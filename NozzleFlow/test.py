from species import *

def test_gibbs_minimizer():
    print("=== Testing Gibbs Minimizer ===")
    spec, elem = get_species_data()
    gibbs = GibbsMinimizer(list(spec.values()), elem)

    # === Simple stoichiometric combustion mixture ===
    b_elem, _ = get_b_elem("H2", "O2", OF_ratio=8.0)
    T = 3000.0
    P = 5e6

    try:
        y_eq = gibbs.solve(T=T, P=P, b_elem=b_elem)
    except Exception as e:
        print("❌ FAILED: Exception during solve:", e)
        return False

    n = np.array(list(y_eq.values()))
    A = np.array([sp.elem_vec for sp in spec.values()]).T
    b_vec = np.array([b_elem[e] for e in elem])

    # === Check 1: Nonnegative species ===
    if np.any(n < 0):
        print("❌ FAILED: Negative mole numbers found")
        return False

    # === Check 2: Element balance ===
    resid = np.linalg.norm(b_vec - A @ n)
    if resid > 1e-6 * np.linalg.norm(b_vec):
        print(f"❌ FAILED: Element balance off (resid={resid:.2e})")
        return False

    # === Check 3: Dominant species sanity ===
    y_sorted = sorted(y_eq.items(), key=lambda kv: kv[1], reverse=True)
    major = [k for k, v in y_sorted[:5] if v > 1e-6]

    if not any(s in major for s in ("H2O", "OH", "H2", "O2")):
        print("❌ FAILED: No reasonable combustion products in top species.")
        print("Top species:", major)
        return False

    print(f"✅ PASS: Converged at T={T:.0f}K, P={P/1e6:.1f}MPa")
    print(f"   Element balance residual = {resid:.2e}")
    print(f"   Dominant species: {major[:5]}")
    return True


if __name__ == "__main__":
    ok = test_gibbs_minimizer()
    if not ok:
        print("Gibbs minimizer failed basic validation.")
