from HeatTransfer.bartz_formulas import bartz_heat_transfer, total_heat_flux
from MachSolver import mach_from_area_ratio as mach_eps
import numpy as np
from NozzleDesign import build_nozzle
import _extra_utils as utils


def isentropic_nozzle_flow(eps, data: dict):
    # Unpack data dictionary
    T0, P0, gamma, R = data['Tc'], data['Pc'], data['gamma'], data['R']

    # Compute mach through geometry
    M = np.array([mach_eps(eps=e, gamma=gamma) for e in eps])
    # Compute static temp
    T = T0 / (1 + (gamma-1)/2 * M**2)
    # Compute static pressure
    P = P0 * (T/T0)**(gamma/(gamma-1))
    # Compute speed of sound at each station
    a = np.sqrt(gamma * R * T)
    # Compute speed of gas
    U = M * a
    # Compute density
    rho = P / (R * T)

    return {"M": M, "U": U, "T": T, "P": P, "rho": rho}


def main_basic(data: dict):
    Pe, Pc, Tc, gamma, size, R, Rt, k, mu = (data["Pe"], data["Pc"], data["Tc"], data["gamma"], data["size"],
                                      data["R"], data["Rt"], data["k"], data["mu"])

    # Build nozzle
    x, y, a = build_nozzle(data=data)

    # Convert to ratio
    a_min = min(a)
    eps: list = a/a_min

    # Isolate subsonic and supersonic with a negative sign (will be zeroed out eventually)
    ind = np.where(eps == 1.0)[0][0]
    eps[:ind] *= -1

    # Solve isentropic relations
    flow: dict = isentropic_nozzle_flow(eps=eps, data=data)
    flows = [flow["M"], flow["U"], flow["T"], flow["P"], flow["rho"]]
    names = ["M", "U", "T", "P", "rho"]
    subnames = [None, None, None, None, None]
    # utils.plot_flow_char(x=x, data=flows, labels=names)

    # Solve for heat transfer
    cp = gamma * R / (gamma - 1)
    q: dict = bartz_heat_transfer(x=x, y=y, cp=cp,
                                  T=flow["T"], M=flow["M"], info=data)

    flows1 = [q["qdot"], (q["T_wi"], q["T_wo"])]
    names1 = ["Heat Flux q", "Temps"]
    subnames1 = [None, ["Inner Temps", "Outer Temps"]]

    # Tc_out is the temperature of coolant leaving the regens and entering the injector
    # Tc_out is also a target value
    # Compute the total heat transfer and mdot
    Tc_out = 350
    Q: dict = total_heat_flux(qdot=q["qdot"], x=x, y=y, cp=cp, Tc_in=q["T_ci"], Tc_out=Tc_out)
    # print(f"Mass flow rate (first pass): {Q["mdot"]:.3f}kg/s")
    print(f"Total heat flux (Q): {Q["Qtotal"]:.2f}W")
    flows2 = [Q["Q"]]
    names2 = ["Total Q"]
    subnames2 = [None]


    flows = flows + flows1 + flows2
    names = names + names1 + names2
    subnames = subnames + subnames1 + subnames2
    utils.plot_flow_chart(x=x, data=flows, labels=names, sublabels=subnames)

    utils.plot_flow_field(x, y, data=q["qdot"], label="Heat Flux")


if __name__ == '__main__':

    # == ENGINE INFO == #
    info = {"Pc": 2.013e6,      # Chamber Pressure [Pa]
            "Pe": 1.01325e5,    # Ambient Pressure (exit) [Pa]
            "Tc": 3200,         # Chamber temp [K]
            "gamma":1.22,       # Hot gas specific heat ratio
            "R": 350,           # Hot gas ideal gas constant
            "mu": 8.617e-4,     # Hot gas dynamic viscosity
            "k":0.5937,         # Wall thermal conductivity
            "Rt": 0.05,         # Throat radius
            "size": 0.8,        # Engine proportion (% of rao nozzle)
            "plots": "no"       # Choice of engine plotting (no, 2D, 3D)
            }

    # == RUN ME == #
    main_basic(data=info)
