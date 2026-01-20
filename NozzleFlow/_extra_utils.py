import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from scipy.interpolate import CubicSpline
from GeometryDesign import view_channel_slices
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def data_at_point(A, B, value):
    """
    Determine a specific value
    :param A: get index of this
    :param B: and find that index here
    :param value: return value at B
    :return:
    """
    idx = np.argmin(np.abs(A - value))
    return B[idx]


def plot_engine(x, y, type="2D"):

    if type == "2D":
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(x, y, color="b")
        ax.plot(x, -y, color="b")

        # Force pyplot-like autoscaling
        ax.set_aspect('equal', adjustable='box')
        ax.autoscale()
        plt.tight_layout()

        ax.set_title("Ideal - Rao Nozzle")
        ax.set_xlabel("Length [m]")
        ax.set_ylabel("Radius [m]")

        ax.grid()
        ax.minorticks_on()
        ax.grid(which='major', linestyle='-', linewidth='0.5')
        ax.grid(which='minor', linestyle=':', linewidth='0.5')

    if type == "3D":
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        phi = np.linspace(0, 2 * np.pi, 250)
        Z, PHI = np.meshgrid(x, phi)
        Y = np.outer(np.ones_like(phi), y) * np.cos(PHI)
        X = np.outer(np.ones_like(phi), y) * np.sin(PHI)

        ax.plot_surface(X, Y, -Z, color="b", edgecolor='g', linewidth=0.5, shade=True)
        # ax.plot_surface(X, Y, -Z, color="b", linewidth=2)
        # ax.axis('equal')
    plt.show()


def plot_flow_field(x, y, data, label, ax=None, cmap='plasma', add_colorbar=True):
    """
    Visualize 1D flow property within a symmetric nozzle contour.

    Parameters
    ----------
    x, y : 1D arrays
        Axial and radial geometry coordinates (upper wall).
    data : 1D array
        Flow property along x (Mach, Pressure, etc.).
    label : str
        Colorbar label.
    mode : int
        1 = discrete PolyCollection bands
        2 = smooth interpolated field
    cmap : str
        Matplotlib colormap name.
    """

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
        created_fig = True

    else:
        fig = ax.figure

    verts, colors = [], []
    for i in range(len(x) - 1):
        verts.append([
            (x[i],   -y[i]),
            (x[i],    y[i]),
            (x[i+1],  y[i+1]),
            (x[i+1], -y[i+1])
        ])
        colors.append(data[i])

    poly = PolyCollection(verts, array=np.array(colors),
                          cmap=cmap, edgecolors='none')
    ax.add_collection(poly)
    ax.plot(x,  y, 'k', lw=1.5)
    ax.plot(x, -y, 'k', lw=1.5)
    ax.set_aspect('equal')
    ax.set_xlabel("Length [m]")
    ax.set_ylabel("Radius [m]")
    if add_colorbar and created_fig:
        fig.colorbar(poly, ax=ax, label=label)

    return poly


def plot_flow_chart(x, flows, labels, sublabels=None, fig=None, axes=None):
    """
    Plots multiple flow characteristics along a nozzle.
    Accepts either single y-arrays or tuples/lists of y-arrays for multi-line subplots.

    :param x:          1D array of axial positions
    :param data:       list of y-arrays or tuples/lists of y-arrays
    :param labels:     list of y-axis labels for each subplot
    :param sublabels:  optional list of sublabels (list of lists/tuples) for multi-line plots
    """
    l = len(flows)
    ncols = 2
    nrows = int(np.ceil(l / ncols))

    if fig is None:
        fig = plt.figure(figsize=(8, 2 * nrows))
        axes = None

    for i in range(l):
        if axes is None:
            ax = fig.add_subplot(nrows, ncols, i+1)
        else:
            ax = axes[i]

        label = labels[i] if i < len(labels) else f"Data {i+1}"

        # Handle tuple/list of arrays -> multiple lines per subplot
        if isinstance(flows[i], (tuple, list)):
            y_group = flows[i]
            # Get sublabels (if provided)
            sl = sublabels[i] if sublabels and i < len(sublabels) else [f"Set {j+1}" for j in range(len(y_group))]
            for y, slbl in zip(y_group, sl):
                ax.plot(x, y, label=slbl)
            ax.legend(fontsize=8)
        else:
            ax.plot(x, flows[i], label=label)

        ax.set_ylabel(label)
        ax.grid(True)
        ax.minorticks_on()
        ax.grid(which='minor', linestyle=':', linewidth='0.5')

    plt.xlabel("Axial Length [m]")
    # plt.tight_layout()


def convert_to_func(x, y, save, filename="nozzle_curve.txt"):
    """
    Exports (x, y) nozzle geometry as a SolidWorks-compatible text file for
    'Curve Through XYZ Points' import.
    """
    # Sort and clean x-values
    sort_idx = np.argsort(x)
    x, y = np.array(x)[sort_idx], np.array(y)[sort_idx]

    tol = 1e-9
    unique_idx = np.diff(x, prepend=-np.inf) > tol
    x, y = x[unique_idx], y[unique_idx]

    if len(x) < 3:
        raise ValueError("Need at least 3 unique x points to export.")

    # Smooth with cubic spline
    spline = CubicSpline(x, y)
    x_dense = np.linspace(x.min(), x.max(), 1000)
    y_dense = spline(x_dense)

    # Z = 0 for 2D profile
    z_dense = np.zeros_like(x_dense)

    # Stack columns: X Y Z
    if save:
        data = np.column_stack((x_dense, y_dense, z_dense))
        np.savetxt(filename, data, fmt="%.6f", delimiter="\t")

        print(f"✅ Saved SolidWorks-compatible curve: {filename} ({len(x_dense)} points)")
        print("→ SolidWorks: Insert > Curve > Curve Through XYZ Points > Select this .txt")
    return x_dense, y_dense


def Moody_plot(eps_dh: float, Re: float):
    """
    Determines the friction factor using the Haaland formula
    """

    inf_sqrt_f = -1.8 * np.log10((eps_dh/3.7)**1.11 + (6.9/Re))
    f = (1/inf_sqrt_f)**2
    return f


def data_display(data: dict):
    # os.system("cls" if os.name == "nt" else "clear")
    frmt                = "{:<50} {:<12.3f} {:<12} {:<12.3f} {:<12} {:<}"
    frmt2               = "{:<50} {:<12} {:<12} {:<12} {:<12} {:<}"
    frmte               = "{:<50} {:<12.3e} {:<12} {:<12.3e} {:<12} {:<}"
    frmtext             = "{:<50} {:<12} {:<12} {:<12.3f} {:<12} {:<}"
    frmtint             = "{:<50} {:<12.3f} {:<12} {:<12} {:<12} {:<}"

    q = data["q"]

    y = data["E"]["y"]
    x = data["E"]["x"]
    eps = data["E"]["aspect_ratio"]
    Lc = data["E"]["Lc"] if data["E"]["Lc"] is not None else 0

    Pc = data["E"]["Pc"]
    Tc = data["E"]["Tc"]
    gamma = data["H"]["gamma"]
    if type(gamma) == np.ndarray:
        gamma = gamma[0]
    R = data["H"]["R"]

    mdot = data["E"]["mdot"]
    exit_vel = data["Flow"]["U"][-1]

    # ============== #
    # == PRINTING == #
    # ============== #
    print("=" * 102, f"{'|':<}")
    print(f"{'ENGINE GEOMETRY':^100} {'|':>3}")
    print("- " * 51, f"{'|':<}")

    print(frmt.format("Throat Diameter", min(y) * 2*100, "cm", min(y)*39.3701*2, "in", "|"))
    print(frmt.format("Exit Velocity", exit_vel, "m/s", exit_vel*3.28084, "ft/s", "|"))
    print(frmt.format("Exit Diameter", y[-1] * 2, "m", y[-1]*2*39.3701, "in", "|"))
    print(frmt.format("Total Force @ SL", exit_vel * mdot / 1e3, "kN", exit_vel * mdot / 1e3 * 224.809, "lbs", "|"))
    print(frmt.format("Total Engine Length", x[-1] - x[0], "m", (x[-1] - x[0])*39.3701, "in", "|"))
    print(frmt.format("Mass Flow Rate", mdot, "kg/s", mdot*2.20462, "lbs/s", "|"))
    of = data["E"]["OF"]
    of_f = mdot / (of + 1)
    of_l = of * mdot / (of + 1)
    print(frmt.format("Fuel Flow Rate", of_f, "kg/s", of_f*2.20462, "lbs/s", "|"))
    print(frmt.format("Ox Flow Rate", of_l, "kg/s", of_l*2.20462, "lbs/s", "|"))
    print(frmtext.format("Expansion Ratio", "", "", np.max(eps), " ", "|"))
    print(frmt.format("Chamber Radius", data["E"]["y"][-1]*1000, "mm", data["E"]["y"][-1]*39.3701, "in", "|"))
    print(frmt.format("Chamber Length", Lc/1000, "mm", Lc*39.3701, "in", "|"))
    print(frmt.format("Characteristic Velocity", data["H"]["cstar"], "m/s", data["H"]["cstar"]*3.28084, "ft/s", "|"))

    print("="*102,f"{'|':<}")
    print(f"{'GAS CONDITIONS':^100} {'|':>3}")
    print("- " * 51, f"{'|':<}")

    P_exit = round(data["Flow"]["P"][-1], 3)
    P_ambient = round(data["E"]["Pe"], 3)
    if P_exit < P_ambient:
        condition = "Over"
    elif P_exit > P_ambient:
        condition = "Under"
    else:
        condition = "Perfect"

    print(frmt.format("Chamber Pressure", Pc/1e6, "MPa", Pc*0.000145038, "psi", "|"))
    print(frmtint.format("Chamber Temperature", Tc, "K", "--", "R", "|"))
    print(frmt.format("Exit Pressure", data["Flow"]["P"][-1] / 1e6, "MPa", data["Flow"]["P"][-1]*0.000145038, "psi", "|"))
    print(frmt.format("Ambient Pressure", P_ambient / 1e6, "MPa", P_ambient*0.000145038, "psi", "|"))
    print(frmt2.format("Expansion Condition", "", "", condition, "", "|"))
    print(frmtext.format("Gamma", "", "", gamma, "", "|"))
    print(frmt.format("Gas Constant (R)", R, "J/kg-K", R*0.0002390057, "Btu/lbs-F", "|"))
    print(frmt.format("Gas Coefficient of Constant Pressure (cp_g)", data["H"]["cp"][1], "J/kg-K", data["H"]["cp"][1]*0.0002390057, "Btu/lbs-F", "|"))
    print(frmtext.format("OF Ratio", "", "", data["E"]["OF"], "", "|"))
    k_gas               = data["H"]["k"]
    print(frmt2.format("Thermal Conductivity", "", "", "", "", "|"))
    print(frmt.format("    Chamber", k_gas[0], "W/(m-K)", k_gas[0]*0.577789, "Btu/hr-ft-F", "|"))
    print(frmt.format("    Throat", k_gas[1], "W/(m-K)", k_gas[1]*0.577789, "Btu/hr-ft-F","|"))
    print(frmt.format("    Exit", k_gas[2], "W/(m-K)", k_gas[2]*0.577789, "Btu/hr-ft-F","|"))
    mu                  = data["H"]["mu"]
    print(frmt2.format("Dynamic Viscosity", "", "", "", "", "|"))
    print(frmte.format("    Chamber", mu[0], "Pa-s", mu[0]*1.450377, "psi-s", "|"))
    print(frmte.format("    Throat", mu[1], "Pa-s", mu[1]*1.450377, "psi-s", "|"))
    print(frmte.format("    Exit", mu[2], "Pa-s", mu[2]*1.450377, "psi-s", "|"))
    Pr                  = data["H"]["Pr"]
    print(frmt2.format("Prandtl Number", "", "", "", "", "|"))
    print(frmtext.format("    Chamber", "", "", Pr[0], "", "|"))
    print(frmtext.format("    Throat", "", "", Pr[1], "", "|"))
    print(frmtext.format("    Exit", "", "", Pr[2], "", "|"))
    print(frmtint.format("Molar Weight", data["H"]["MW"], "g/mol", "--", "lbm/lbmol", "|"))


    print("=" * 102, f"{'|':<}")
    print(f"{'Generated Ideal Cooling Geometry Based On H&H':^100} {'|':>3}")
    print("- " * 51, f"{'|':<}")
    print(frmtint.format("Max Wall Temp Used", data["W"]["max_wall_temp"], "K", "", "R", "|"))
    print(frmt2.format("Geometry Type", "", "", data["C"]["Type"].title(), "",  "|"))
    print(frmt.format("Pressure Drop Through Channels", data["C"]["dP"]/1e6, "MPa", data["C"]["dP"]*0.000145037, "psi", "|"))
    print(frmt.format("Fuel Inlet Pressure", data["F"]["StartingPressure"]/1e6, "MPa", data["F"]["StartingPressure"]*0.000145037, "psi", "|"))

    if data["C"]["Type"].lower() == "circle":
        print(frmtext.format("Number of Channels", "", "", data["C"]["num_ch"], "", "|"))
        print(frmt.format("Inner Diameter", data["C"]["height"] * 1000, "mm", "|"))
        print(frmt.format("Outer Diameter", data["C"]["spacing"] * 1000, "mm", "|"))
        print(frmt.format("Wall Thickness", data["C"]["wall_thickness"] * 1000, "mm", "|"))
        print(frmte.format("Channel Area", np.pi * data["C"]["height"] ** 2 / 4, "m", "|"))
        print(frmt.format("Mass Flow Per Channel", data["F"]["mdot"] / data["C"]["num_ch"] * 1000, "g/s", "|"))
        print(frmt.format("Coolant Bulk Temp", data["C"]["throat_bulk_temp"], "K", "|"))

    elif data["C"]["Type"].lower() == "square":
        print(frmtext.format("Number of Channels", "", "", data["C"]["num_ch"], "", "|"))
        print(frmt2.format("Edge Width", "", "", "", "", "|"))
        print(frmt.format("     Throat", data["C"]["height"] * 1000, "mm", data["C"]["height"]*39.3701, "in", "|"))
        print(frmt.format('     Chamber', data["C"]["width_arr"][0] * 1000, "mm", data["C"]["width_arr"][0]*39.3701, "in", "|"))
        print(frmt.format("    Exit", data["C"]["width_arr"][-1]*1000, "mm", data["C"]["width_arr"][-1]*39.3701, "in", "|"))
        print(frmt2.format("Edge Depth", "", "", "", "", "|"))
        print(frmt.format("     Throat", data["C"]["spacing"] * 1000, "mm", data["C"]["spacing"]*39.3701, "in", "|"))
        print(frmt.format('     Chamber', data["C"]["depth_arr"][0] * 1000, "mm", data["C"]["depth_arr"][0]*39.3701, "in", "|"))
        print(frmt.format("    Exit", data["C"]["depth_arr"][-1]*1000, "mm", data["C"]["depth_arr"][-1]*39.3701, "in", "|"))
        print(frmt.format("Fin Thickness", data["C"]["spacing"] * 1000, "mm", data["C"]["spacing"]*39.3701, "in", "|"))
        print(frmt.format("Wall Thickness", data["W"]["thickness"] * 1000, "mm", data["W"]["thickness"]*39.3701, "in", "|"))
        print(frmt.format("Channel Area", data["C"]["spacing"] ** 2, "m2", data["C"]["spacing"] ** 2*1550, "in2", "|"))
        print(frmt.format("Mass Flow Per Channel", data["F"]["mdot"] / data["C"]["num_ch"] * 1000, "g/s", data["F"]["mdot"] / data["C"]["num_ch"]*2.20462, "lb/s", "|"))
        print(frmtint.format("Coolant Bulk Temp", data["C"]["throat_bulk_temp"], "K", "", "R", "|"))





    if q is not None:

        max_wall_temp_x = data_at_point(A=q["T_wall_gas"], B=x, value=np.max(q["T_wall_gas"]))
        max_wall_temp   = np.max(q["T_wall_gas"])

        print("=" * 102, f"{'|':<}")
        print(f"{'HEAT DATA':^100} {'|':>3}")
        print("- " * 51, f"{'|':<}")

        print(frmtint.format("Maximum Wall Temp", max_wall_temp, "K", "", "R", "|"))
        print(frmt.format("at ... from throat", max_wall_temp_x * 1000, "mm", max_wall_temp_x*39.3701, "in", "|"))
        print(frmtint.format("Maximum Coolant Temp", np.max(q["T_cool"]), "K", "", "R", "|"))
        print(frmt.format("Coolant Critical Pressure", data["F"]["State"].p_critical()/1e6, "MPa", data["F"]["State"].p_critical()*0.000145038, "psi", "|"))
        print(frmtint.format("Coolant Temp at Throat", data_at_point(A=data["E"]["x"], B=q["T_cool"], value=0), "K", "", "R", "|"))
        print(frmt.format("Regen Channel Pressure Drop", (q["P_c"][-1] - q["P_c"][0])/1000, "kPa", (q["P_c"][-1] - q["P_c"][0])*0.000145038, "psi", "|"))
        print(frmt.format("Injector Pressure Drop", data["Injector"]["dP"] - data["E"]["Pc"], "Pa", (data["Injector"]["dP"] - data["E"]["Pc"])*0.000145038, "psi", "|"))
        print(frmt.format("Recommended Inlet Pressure", data["F"]["StartingPressure"]/1e6, "MPa", data["F"]["StartingPressure"]*0.000145038, "psi", "|"))
        # if np.max(q["T_cool"]) == data["F"]["T_max"]:
        #     print(frmt2.format("The coolant exceeded the thermally stable temperature region", "","","|"))
        #     print(frmt.format("The coolant was therefor clamped to", data["F"]["T_max"], "K", "|"))
        # print(frmt.format("Average Heat Transfer Coefficient (hot gas)", np.mean(q["h_hg"]) / 1000, "kW/m^2-K", "|"))
        print(frmt.format("Maximum Heat Transfer Coefficient (hot gas)", np.max(q["h_hg"]) / 1000, "kW/m^2-K", np.max(q["h_hg"]*0.176228), "Btu/hr/ft2", "|"))
        # print(frmt.format("Maximum Heat Transfer Coefficient (wall->coolant", max(q["h_wc"]) / 1000, "kW/m^2-K", "|"))
        print(frmt.format("Maximum hHat Flux (hot gas)", np.max(q["qpp_hg"]) / 1000, "kW/m2",
                          np.max(q["h_hg"] * 0.176228), "Btu/hr/ft2", "|"))

        # print(frmt.format("Average Heat Rate (Qdot)", np.max(q["Q/_dot"]), "W", "|"))
        # print(frmt.format("Total Heat rate (Qdot)", sum(q["Q_dot"]), "W", "|"))

        print("=" * 102, f"{'|':<}")
        print(f"{'---':^100} {'|':>3}")
        print("- " * 51, f"{'|':<}")


        melting_point = data["W"]["solidus"]
        if max_wall_temp > melting_point:
            excess = melting_point - max_wall_temp
            print(
                f"WARNING : Maximum wall temp exceeds the melting point of {data["W"]["Type"]} by {abs(excess):.2f} K")
            percent = abs(melting_point - max_wall_temp) / max_wall_temp * 100
            print(f"WARNING : This is a {percent:.2f}% error")


        print(f"WARNING : All pressure drops and values are estimations, they also do not account for channel inlet conditions")


def plot_info(data: dict):

    energy_plot     = data["Display"]["EnergyPlot"]
    flow_plot       = data["Display"]["FlowPlot"]
    contour_plot    = data["Display"]["ContourPlot"]
    channel_plot    = data["Display"]["ChannelPlot"]

    if not energy_plot and not flow_plot and not contour_plot:
        print("Cancelling plot, incorrectly called")
        return

    q               = data["q"]
    x               = data["E"]["x"]
    y               = data["E"]["y"]

    flows           = []
    names           = []
    subnames        = []

    if energy_plot and data["q"] is not None:
        # print(data["q"])
        flows       += [(q["h_hg"], q["h_wc"]),
                               (q["T_wall_gas"], q["T_wall_coolant"], q["T_aw"]),
                               (q["R_hg_w"], q["R_w_w"], q["R_w_c"]),
                               q["Q_dot"]/1000,
                               q["qpp_hg"],
                               q["T_cool"]
                               ]
        names       += ["Heat Transfer Coefficients",
                               "Wall Temps",
                               "Wall Resistances",
                               "Q_dot",
                               "Heat flux",
                               "Coolant Temp"
                               ]
        subnames    += [("Gas-Wall", "Wall-Coolant"),
                               ("Wall-Gas", "Wall-Coolant", "Adiabatic Wall"),
                               ("Wall-Gas", "Wall-Wall", "Wall-Coolant"),
                               None,
                               None,
                               None
                               ]

        # flows += [(q["T_wall_gas"], q["T_wall_coolant"]), q["T_cool"], q["P_c"]]
        # names += ["Wall Temps", "Coolant Temp", "Coolant Pressure"]
        # subnames += [("Inner Wall", "Outer Wall"), None, None]
    if flow_plot:
        flows += [data["Flow"]["M"], data["Flow"]["U"], data["Flow"]["T"], data["Flow"]["P"], data["Flow"]["rho"]]
        names += ["M", "U", "T", "P", "rho"]
        subnames += [None, None, None, None, None]

    nplots = len(flows)

    fig = plt.figure(figsize = (16, 10))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.25)
    left = outer[0].subgridspec(3, 1, height_ratios=[3.5,3.5,0.5], hspace=0.15)

    ax_field = fig.add_subplot(left[0])
    ax_slice = fig.add_subplot(left[1])
    ax_slider = fig.add_subplot(left[2])

    right_rows = int(np.ceil(nplots/2))

    right = outer[1].subgridspec(right_rows, 2, hspace=0.35, wspace=0.25)

    axes = []
    for i in range(nplots):
        r = i//2
        c = i % 2
        axes.append(fig.add_subplot(right[r, c]))

    if contour_plot and data["q"] is not None:
        contour = plot_flow_field(x=x, y=y, data=q["T_wall_gas"], label="Inner Wall Temp", ax=ax_field)
        fig.colorbar(contour, ax=ax_field, label="Inner Wall Temp")

    if channel_plot:
        slider = view_channel_slices(data=data, fig=fig, ax=ax_slice, ax_slider=ax_slider)

    plot_flow_chart(x=x, flows=flows, labels=names, sublabels=subnames, fig=fig, axes=axes)

    plt.subplots_adjust(left=0.05, right=0.97, top=0.96, bottom=0.06)
    plt.show()



class ConvergencePlot:
    def __init__(self, title="Convergence", ylabel="Residuals", yscale='log'):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_title(title)
        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel(ylabel)
        self.ax.set_yscale(yscale)
        self.ax.grid(True)
        self.line, = self.ax.plot([], [], "o-", color="tab:blue")
        self.xs, self.ys = [], []

    def update(self, iteration, residual):
        self.xs.append(iteration)
        self.ys.append(abs(residual) if abs(residual) > 0 else 1e-12)
        self.line.set_data(self.xs, self.ys)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.pause(0.05)

    def close(self):
        plt.ioff()
        plt.close(self.fig)




