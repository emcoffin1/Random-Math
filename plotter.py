import matplotlib.pyplot as plt
import pandas as pd

# ===========================
# Load Data
# ===========================
one   = pd.read_excel("NozzleJet.xlsx", sheet_name=0, usecols="D:M",  skiprows=1)
two   = pd.read_excel("NozzleJet.xlsx", sheet_name=0, usecols="O:X",  skiprows=1)
three = pd.read_excel("NozzleJet.xlsx", sheet_name=0, usecols="Z:AI", skiprows=1)
four  = pd.read_excel("NozzleJet.xlsx", sheet_name=0, usecols="AK:AT", skiprows=1)
gen   = pd.read_excel("NozzleJet.xlsx", sheet_name=0, usecols="AW:BF", skiprows=1)

# Extract columns
one_v   = one["Velocity (mean) m/s"].dropna()
two_v   = two["Velocity (mean) m/s.1"].dropna()
three_v = three["Velocity (mean) m/s.2"].dropna()
four_v  = four["Velocity (mean) m/s.3"].dropna()

one_r   = one["r from center"].dropna()
two_r   = two["r from center.1"].dropna()
three_r = three["r from center.2"].dropna()
four_r  = four["r from center.3"].dropna()

# ===========================
# Plot 1: Velocity vs Radius
# ===========================
def plot_velocity_vs_radius(v_lists, r_lists, titles):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharex=True, sharey=True)

    # Determine global axis limits
    vmin = min(v.min() for v in v_lists)
    vmax = max(v.max() for v in v_lists)
    rmin = min(r.min() for r in r_lists)
    rmax = max(r.max() for r in r_lists)

    for ax, v, r, title in zip(axes, v_lists, r_lists, titles):
        ax.plot(v, r, marker='o', markersize=3)
        ax.set_title(title)
        ax.set_xlabel("Velocity (m/s)")
        ax.grid(True, linestyle='--', alpha=0.5)

        ax.set_xlim(vmin, vmax)
        ax.set_ylim(rmin, rmax)

    axes[0].set_ylabel("Radius r (m)")
    plt.suptitle("Jet Velocity Profiles at Different x/D")
    plt.tight_layout()
    plt.show()


# ===========================
# Plot 2: U/U_m vs r/D
# ===========================
one_u_um   = one["U/U_m"].dropna()
two_u_um   = two["U/U_m.1"].dropna()
three_u_um = three["U/U_m.2"].dropna()
four_u_um  = four["U/U_m.3"].dropna()

one_r_d   = one["r/D"].dropna()
two_r_d   = two["r/D.1"].dropna()
three_r_d = three["r/D.2"].dropna()
four_r_d  = four["r/D.3"].dropna()


def plot_u_over_um_vs_r_over_d(u_lists, r_lists, titles):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))

    for ax, u, r, title in zip(axes, u_lists, r_lists, titles):
        ax.plot(u, r, 'b')
        ax.set_title(title)
        ax.set_xlabel("U / U_m")
        ax.set_xlim(0, 1.1)
        ax.grid(True)

    axes[0].set_ylabel("r/D")
    plt.suptitle("Normalized Jet Profiles (U/U_m vs r/D)")
    plt.tight_layout()
    plt.show()



gen_um_u0 = gen["Um/U0"].dropna()
gen_xd = gen["x/D"].dropna()

gen_q_q0 = gen["Q/Q0"].dropna()

gen_r_2 = gen["r half / D"].dropna()


# ===========================
# Call Both Plots
# ===========================
titles = ["x/D = 0.1", "x/D = 1.0", "x/D = 2.5", "x/D = 5.0"]

plot_velocity_vs_radius(
    [one_v, two_v, three_v, four_v],
    [one_r, two_r, three_r, four_r],
    titles
)

plot_u_over_um_vs_r_over_d(
    [one_u_um, two_u_um, three_u_um, four_u_um],
    [one_r_d, two_r_d, three_r_d, four_r_d],
    titles
)


plt.plot(gen_xd, gen_um_u0)
plt.xlabel("x/D")
plt.ylabel("Um/U0")
plt.title("x/D vs Um/U0")
plt.show()


plt.plot(gen_xd, gen_q_q0)
plt.xlabel("x/D")
plt.ylabel("Q/Q0")
plt.title("x/D vs Q/Q0")
plt.show()

plt.plot(gen_xd, gen_r_2)
plt.xlabel("x/D")
plt.ylabel("r(1/2) / D")
plt.title("x/D vs r(1/2) / D")
plt.show()