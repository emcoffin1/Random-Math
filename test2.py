"""
CEA → ANSYS (Fluent/CFX) handoff script
- Most accurate + stable workflow: use CEA for composition + state, ANSYS for thermo/transport
- Outputs: chamber/throat/exit T, MW, gamma, cp, and species mass fractions (frozen + optional equilibrium)
"""

from rocketcea.cea_obj import CEA_Obj
import json
from dataclasses import dataclass, asdict

# -------------------------
# USER INPUTS
# -------------------------
Pc_psi = 305.0          # chamber pressure [psi]
MR = 1.8                # mixture ratio O/F
eps = 20.0              # expansion ratio Ae/At (CHANGE THIS to your nozzle)
fuel = "RP-1"
oxidizer = "LOX"

# Filtering: ANSYS hates 1e-12 noise in species lists
MIN_Y = 1e-6            # drop species below this mass fraction

# -------------------------
# CONSTANTS / UNIT CONVERSIONS
# -------------------------
PSI_TO_PA = 6894.757293168
R_TO_K = 5.0 / 9.0
BTU_PER_LBM_R_TO_J_PER_KG_K = 4186.8  # exact enough for this use

Pc_Pa = Pc_psi * PSI_TO_PA


@dataclass
class State:
    name: str
    P_Pa: float | None
    T_K: float
    gamma: float
    MW_kg_per_kmol: float
    cp_J_per_kgK: float

def extract_species_dict(species_return):
    """
    RocketCEA may return:
      - dict
      - (dict, ...)
    This function always returns the dict.
    """
    if isinstance(species_return, dict):
        return species_return

    if isinstance(species_return, (tuple, list)):
        for item in species_return:
            if isinstance(item, dict):
                return item

    raise TypeError(
        f"Could not extract species dict from type: {type(species_return)}"
    )



def normalize_and_filter(species_mass_fracs, min_y):
    """
    Fully hardened against RocketCEA weirdness:
    - tuple vs dict returns
    - scalar vs [scalar] values
    """
    species_mass_fracs = extract_species_dict(species_mass_fracs)

    cleaned = {}

    for k, v in species_mass_fracs.items():
        # Handle list-wrapped scalars
        if isinstance(v, (list, tuple)):
            if len(v) == 0:
                continue
            v = v[0]

        v = float(v)

        if v >= min_y:
            cleaned[k] = v

    s = sum(cleaned.values())
    if s <= 0.0:
        raise ValueError("Species fractions summed to zero after filtering.")

    return {k: v / s for k, v in cleaned.items()}




def cea_state_and_composition(cea: CEA_Obj, location: str, frozen: bool):
    """
    location: "chamber", "throat", "exit"
    frozen: True/False
    Returns: State, species mass fraction dict
    """
    frozen_flag = 1 if frozen else 0

    if location == "chamber":
        T_R = cea.get_Tcomb(Pc=Pc_psi, MR=MR)
        gamma = cea.get_Chamber_MolWt_gamma(Pc=Pc_psi, MR=MR, eps=eps)[1]
        MW = cea.get_Chamber_MolWt_gamma(Pc=Pc_psi, MR=MR, eps=eps)[0]
        cp_Btu = cea.get_Chamber_Transport(Pc=Pc_psi, MR=MR, eps=eps, frozen=frozen_flag)[0]
        # Chamber species at Pc
        spec = cea.get_SpeciesMassFractions(
            Pc=Pc_psi,
            MR=MR,
            eps=eps,
            frozen=frozen_flag
        )

        st = State(
            name=f"{location} ({'frozen' if frozen else 'equil'})",
            P_Pa=Pc_Pa,
            T_K=T_R * R_TO_K,
            gamma=float(gamma),
            MW_kg_per_kmol=float(MW),  # numerically same as kg/kmol
            cp_J_per_kgK=float(cp_Btu) * BTU_PER_LBM_R_TO_J_PER_KG_K,
        )
        return st, normalize_and_filter(spec, MIN_Y)

    if location == "throat":
        T_R = cea.get_Tcomb(Pc=Pc_psi, MR=MR)
        gamma = cea.get_Throat_MolWt_gamma(Pc=Pc_psi, MR=MR, eps=eps)[1]
        MW = cea.get_Throat_MolWt_gamma(Pc=Pc_psi, MR=MR, eps=eps)[0]
        cp_Btu = cea.get_Throat_Transport(Pc=Pc_psi, MR=MR, eps=eps, frozen=frozen_flag)[0]
        # Chamber species at Pc
        spec = cea.get_SpeciesMassFractions(
            Pc=Pc_psi,
            MR=MR,
            eps=eps,
            frozen=frozen_flag
        )

        st = State(
            name=f"{location} ({'frozen' if frozen else 'equil'})",
            P_Pa=None,  # throat static P is not a single input here; ANSYS solves it
            T_K=T_R * R_TO_K,
            gamma=float(gamma),
            MW_kg_per_kmol=float(MW),
            cp_J_per_kgK=float(cp_Btu) * BTU_PER_LBM_R_TO_J_PER_KG_K,
        )
        return st, normalize_and_filter(spec, MIN_Y)

    if location == "exit":
        # Exit state depends on eps
        T_R = cea.get_Tcomb(Pc=Pc_psi, MR=MR)
        gamma = cea.get_exit_MolWt_gamma(Pc=Pc_psi, MR=MR, eps=eps)[1]
        MW = cea.get_exit_MolWt_gamma(Pc=Pc_psi, MR=MR, eps=eps)[0]
        cp_Btu = cea.get_Exit_Transport(Pc=Pc_psi, MR=MR, eps=eps, frozen=frozen_flag)[0]
        # Chamber species at Pc
        spec = cea.get_SpeciesMassFractions(
            Pc=Pc_psi,
            MR=MR,
            eps=eps,
            frozen=frozen_flag
        )

        st = State(
            name=f"{location} ({'frozen' if frozen else 'equil'})",
            P_Pa=None,
            T_K=T_R * R_TO_K,
            gamma=float(gamma),
            MW_kg_per_kmol=float(MW),
            cp_J_per_kgK=float(cp_Btu) * BTU_PER_LBM_R_TO_J_PER_KG_K,
        )
        return st, normalize_and_filter(spec, MIN_Y)

    raise ValueError(f"Unknown location: {location}")


def main():
    cea = CEA_Obj(oxName=oxidizer, fuelName=fuel, useFastLookup=False)

    out = {
        "inputs": {
            "Pc_psi": Pc_psi,
            "Pc_Pa": Pc_Pa,
            "MR": MR,
            "eps_Ae_At": eps,
            "oxidizer": oxidizer,
            "fuel": fuel,
            "species_filter_min_massfrac": MIN_Y,
        },
        "recommendation_for_ANSYS": {
            "model": "Frozen products mixture; NO reactions",
            "density": "Ideal gas",
            "thermo": "Use built-in species NASA polynomials in ANSYS (cp(T), h(T) automatic)",
            "transport": "Use ANSYS mixture-averaged transport; avoid custom Sutherland fits initially",
            "inlet": "Set T0 or T (depending on your BC) and species mass fractions from chamber-frozen",
        },
        "states": {},
        "species_mass_fractions": {},
    }

    for frozen in (True, False):
        tag = "frozen" if frozen else "equilibrium"
        out["states"][tag] = {}
        out["species_mass_fractions"][tag] = {}

        for loc in ("chamber", "throat", "exit"):
            st, spec = cea_state_and_composition(cea, loc, frozen=frozen)
            out["states"][tag][loc] = asdict(st)
            out["species_mass_fractions"][tag][loc] = spec

    # Write JSON handoff
    with open("cea_ansys_handoff.json", "w") as f:
        json.dump(out, f, indent=2)

    # Print the most-used block for ANSYS inlet: chamber frozen
    inlet_species = out["species_mass_fractions"]["frozen"]["chamber"]
    inlet_state = out["states"]["frozen"]["chamber"]

    print("\n=== ANSYS INLET RECIPE (CHAMBER, FROZEN) ===")
    print(f"Pc = {Pc_Pa:.3f} Pa  (from {Pc_psi} psi)")
    print(f"T  = {inlet_state['T_K']:.3f} K")
    print(f"MW = {inlet_state['MW_kg_per_kmol']:.6f} kg/kmol")
    print(f"gamma = {inlet_state['gamma']:.6f}")
    print(f"cp = {inlet_state['cp_J_per_kgK']:.3f} J/kg-K")
    print("\nSpecies mass fractions (sum=1):")
    for k, v in sorted(inlet_species.items(), key=lambda kv: -kv[1]):
        print(f"  {k:>8s} : {v:.8f}")

    print("\nWrote: cea_ansys_handoff.json")


if __name__ == "__main__":
    main()
