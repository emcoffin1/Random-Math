from pandas.core.config_init import pc_width_doc
from rocketcea.cea_obj import CEA_Obj
import matplotlib.pyplot as plt
import numpy as np

CEA = CEA_Obj(fuelName="RP1", oxName="LOX")
min = 0.25
max = 1.8
At = 1.25e-3
exit_rat = 4.0
def get_gamma(pc, of_s):
    gams = []
    for i in of_s:
        _, gam = CEA.get_Chamber_MolWt_gamma(Pc=pc, MR=i)
        gams.append(gam)
    return gams

def plot_gam(gams, ofs, fuel):

    fix, ax = plt.subplots()
    ax.plot(ofs, gams)
    ax.set_xlabel("OF")
    ax.set_ylabel("Gamma")

    def of_to_lox(of):
        return of * fuel
    def lox_to_of(lox):
        return lox/fuel

    secax = ax.secondary_xaxis(
        'top',
        functions=(of_to_lox, lox_to_of),
    )
    secax.set_xlabel("LOX Flow Rate")
    # ax.axvline(1.8, color='k', linestyle='--')
    ax.grid(True)
    plt.show()

def convert_chamb_pres(pc, total, OF):
    chamb_pres = []
    pc_new = pc

    for i, enum in enumerate(OF):
        # Lets iterate through to get cstar per step
        for j in range(50):
            cstar = CEA.get_Cstar(Pc=pc_new, MR=enum) * 0.3048

            chamb = cstar * total[i] / At

            pc_dif = chamb - pc_new

            if pc_dif < 1e-3:
                chamb_pres.append(chamb)
                break
            else:
                pc_new = chamb
    return chamb_pres

def plot_chamber_pressure(cham_pres, flow_rate):
    plt.plot(flow_rate, cham_pres)
    plt.xlabel("Flow Rate")
    plt.ylabel("Chamber Pressure")
    plt.grid()
    plt.axhline(2.1e6, linestyle='--', color='k')
    plt.show()

def get_thrust(total, pc, of):
    vels = []
    for i in range(len(total)):
        a = CEA.get_SonicVelocities(Pc=pc[i], MR=of[i], eps=4)[2]*0.3048
        mach = CEA.get_MachNumber(Pc=pc[i], MR=of[i], eps=4)
        vels.append(a*mach)
    thrusts = total*vels
    return thrusts

def plot_thrusts(of, thrust):
    fig, ax = plt.subplots()
    ax.plot(of, thrust)
    ax.set_xlabel("OF")
    ax.set_ylabel("Thrust(N)")
    ax.set_title("Thrust Estimation for Varying OF Ratio")

    def lbs_from_N(N):
        return N*0.224809
    def N_from_lbs(lbs):
        return lbs*4.44822

    secax = ax.secondary_yaxis(
        'right',
        functions = (lbs_from_N, N_from_lbs),
    )
    secax.set_xlabel("Thrust(lbf)")
    ax.grid(True)
    plt.show()


if __name__ == "__main__":
    fuel_rate = 0.536
    of_s = np.linspace(min, max, 30)
    # We need to adjust the fuel flow rate
    gams = get_gamma(pc=2.1e6, of_s=of_s)
    lox_rate = of_s * fuel_rate
    chamb_pres = convert_chamb_pres(pc=2.1e6, total=lox_rate+fuel_rate, OF=of_s)
    thrusts = get_thrust(total=fuel_rate+lox_rate, pc=chamb_pres, of=of_s)

    # plot_gam(gams=gams, ofs=of_s, fuel=fuel_rate)
    # plot_chamber_pressure(chamb_pres, flow_rate=fuel_rate+lox_rate)
    plot_thrusts(of=fuel_rate+lox_rate, thrust=thrusts)

