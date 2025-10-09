import re
import numpy as np

def elem_counts(formula):
    """
    Parse a chemical formula like 'C12H26O' into a dict of element counts.
    """
    tokens = re.findall(r'([A-Z][a-z]*)(\d*)', formula)
    counts = {}
    for elem, num in tokens:
        counts[elem] = counts.get(elem, 0) + int(num) if num else counts.get(elem, 0) + 1
    return counts


def get_b_elem(fuel_formula, oxidizer_formula="O2", OF_ratio=3.5):
    """
    Compute b_elem vector [C, H, O, N] from a fuel and oxidizer formula and O/F ratio (mass basis).
    Default oxidizer: O2 (pure oxygen).
    """

    # Universal gas constant (for molecular weights)
    atomic_wt = {'C': 12.011, 'H': 1.008, 'O': 15.999, 'N': 14.007}

    # --- Step 1: parse formulas into element dicts
    fuel_elems = elem_counts(fuel_formula)
    ox_elems   = elem_counts(oxidizer_formula)

    # --- Step 2: compute molecular weights
    def MW(comp):
        return sum(atomic_wt[k]*v for k,v in comp.items())
    MW_fuel = MW(fuel_elems)
    MW_ox   = MW(ox_elems)

    # --- Step 3: compute molar ratio of oxidizer to fuel
    # O/F = (m_ox/m_fuel)
    mol_ratio = (OF_ratio * MW_fuel) / MW_ox   # mol oxidizer per mol fuel

    # --- Step 4: total atom inventory (fuel + oxidizer)
    total = {'C': 0, 'H': 0, 'O': 0, 'N': 0}
    for elem, n in fuel_elems.items():
        total[elem] += n
    for elem, n in ox_elems.items():
        total[elem] += mol_ratio * n

    # --- Step 5: build b_elem vector in consistent order
    # b_elem = np.array([total['C'], total['H'], total['O'], total['N']])
    b_elem = {"C": total["C"], "H": total["H"], "O": total["O"], "N": total["N"]}
    return b_elem, total

