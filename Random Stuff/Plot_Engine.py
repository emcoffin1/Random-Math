import numpy as np
import matplotlib.pyplot as plt

def plot_xy_txt(
    filepath: str,
    mirror_y: bool = False,
    title: str = "Engine Geometry",
    xlabel: str = "x",
    ylabel: str = "y"
):
    """
    Plots an x,y .txt file with equal axis scaling.

    Parameters
    ----------
    filepath : str
        Path to the .txt file containing x y columns
    mirror_y : bool
        If True, mirrors geometry about the x-axis (useful for nozzles)
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    """

    # Load data
    data = np.loadtxt(filepath)
    x = data[:, 0]
    y = data[:, 1]

    # Plot
    fig, ax = plt.subplots()
    ax.plot(x, y, label="Upper contour")

    if mirror_y:
        ax.plot(x, -y, label="Lower contour")

    # Equal scaling is critical for geometry correctness
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.show()


def add_zero_z_column(input_file, output_file):
    # Load x, y columns
    data = np.loadtxt(input_file)

    if data.shape[1] != 2:
        raise ValueError("Input file must have exactly two columns (x y)")

    # Create z column of zeros
    z = np.zeros((data.shape[0], 1))

    # Stack x, y, z
    data_xyz = np.hstack((data, z))

    # Save back to txt
    np.savetxt(
        output_file,
        data_xyz,
        fmt="%.6f",
        delimiter=" "
    )

# Example usage
add_zero_z_column("nozzle_curve_rpa.txt", "nozzle_curve_rpa.txt")


# plot_xy_txt(filepath="nozzle_curve_rpa.txt", mirror_y=True)