import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from scipy.interpolate import CubicSpline


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


def plot_flow_field(x, y, data, label, mode=1, cmap='plasma'):
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

    if mode == 1:
        verts, colors = [], []
        for i in range(len(x) - 1):
            verts.append([
                (x[i],   -y[i]),
                (x[i],    y[i]),
                (x[i+1],  y[i+1]),
                (x[i+1], -y[i+1])
            ])
            colors.append(data[i])

        fig, ax = plt.subplots(figsize=(10, 3))
        poly = PolyCollection(verts, array=np.array(colors),
                              cmap=cmap, edgecolors='none')
        ax.add_collection(poly)
        ax.plot(x,  y, 'k', lw=1.5)
        ax.plot(x, -y, 'k', lw=1.5)
        ax.set_aspect('equal')
        ax.set_xlabel("Length [m]")
        ax.set_ylabel("Radius [m]")
        fig.colorbar(poly, ax=ax, label=label)
        plt.tight_layout()
        plt.show()

    else:
        # create a physical grid for plotting
        nx, ny = 1000, 1000
        X, Y = np.meshgrid(
            np.linspace(x.min(), x.max(), nx),
            np.linspace(-y.max(), y.max(), ny)
        )

        # interpolate flow property to the grid along x
        val_field = np.interp(X[0, :], x, data)
        val_grid = np.tile(val_field, (ny, 1))

        # interpolate the upper/lower wall coordinates for each X column
        y_upper = np.interp(X[0, :], x, y)
        y_lower = -y_upper

        # build mask for inside/outside nozzle
        mask = (Y < y_lower) | (Y > y_upper)
        val_grid[mask] = np.nan

        fig, ax = plt.subplots()
        img = ax.pcolormesh(X, Y, val_grid, shading='auto', cmap=cmap)
        ax.plot(x, y, 'k', lw=1.5)
        ax.plot(x, -y, 'k', lw=1.5)
        ax.set_aspect('equal')
        ax.set_xlabel("Length [m]")
        ax.set_ylabel("Radius [m]")
        fig.colorbar(img, ax=ax, label=label)
        fig.suptitle(f"{label} Through Nozzle")
        plt.show()
        plt.show()


def plot_flow_chart(x, data, labels, sublabels=None):
    """
    Plots multiple flow characteristics along a nozzle.
    Accepts either single y-arrays or tuples/lists of y-arrays for multi-line subplots.

    :param x:          1D array of axial positions
    :param data:       list of y-arrays or tuples/lists of y-arrays
    :param labels:     list of y-axis labels for each subplot
    :param sublabels:  optional list of sublabels (list of lists/tuples) for multi-line plots
    """
    l = len(data)
    ncols = 2
    nrows = int(np.ceil(l / ncols))
    fig = plt.figure(figsize=(8, 2 * nrows))

    for i in range(l):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        label = labels[i] if i < len(labels) else f"Data {i+1}"

        # Handle tuple/list of arrays -> multiple lines per subplot
        if isinstance(data[i], (tuple, list)):
            y_group = data[i]
            # Get sublabels (if provided)
            sl = sublabels[i] if sublabels and i < len(sublabels) else [f"Set {j+1}" for j in range(len(y_group))]
            for y, slbl in zip(y_group, sl):
                ax.plot(x, y, label=slbl)
            ax.legend(fontsize=8)
        else:
            ax.plot(x, data[i], label=label)

        ax.set_ylabel(label)
        ax.grid(True)
        ax.minorticks_on()
        ax.grid(which='minor', linestyle=':', linewidth='0.5')

    plt.xlabel("Axial Length [m]")
    plt.tight_layout()
    plt.show()


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



def moles_to_mass(y, species):
    # Initialize
    denom = 0.0

    # Create denominator for sum(moles * MW)
    for k, v in y.items():
        denom += v * species[k].mw
    # Clip to prevent a random 0
    denom = max(denom, 1e-30)

    # Re-create dict using m_species / m_tot
    w: dict = {k: (y[k] * species[k].mw / denom) for k in y}
    return w

def mass_to_moles(Y, species):
    # Create numerator in form of mass / MW
    numer = {k: (Y[k] / species[k].mw) for k in Y}

    # Create denom which is a sum of the numerators
    denom = sum(numer.values())
    # And clip
    denom = max(denom, 1.0e-30)

    # Reform dictionary in numer/denom
    # mol / mol_tot
    w = {k: (numer[k] / denom) for k in numer}
    return w




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




