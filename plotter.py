import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

one = pd.read_excel("drags2.xlsx", sheet_name=0, usecols="F:N").dropna()
two = pd.read_excel("drags2.xlsx", sheet_name=0, usecols="T:AB").dropna()
three = pd.read_excel("drags2.xlsx", sheet_name=0, usecols="AH:AP").dropna()
four = pd.read_excel("drags2.xlsx", sheet_name=0, usecols="AV:BD").dropna()

bl_x = pd.read_excel("drags2.xlsx", sheet_name=0, usecols="BI").dropna()
bl_h = pd.read_excel("drags2.xlsx", sheet_name=0, usecols="BJ").dropna()
sf_shear = pd.read_excel("drags2.xlsx", sheet_name=0, usecols="BK").dropna()
sf_re = pd.read_excel("drags2.xlsx", sheet_name=0, usecols="BL").dropna()
sf_mom = pd.read_excel("drags2.xlsx", sheet_name=0, usecols="BM").dropna()

print(one.columns.tolist())
print(two.columns.tolist())
print(three.columns.tolist())
print(four.columns.tolist())

# f_degree = one["Degree"]
one_v = one["Local Velocity"]
two_v = two["Local Velocity.1"]
three_v = three["Local Velocity.2"]
four_v = four["Local Velocity.3"]

one_h = one["Elevation"]
two_h = two["Elevation.1"]
three_h = three["Elevation.2"]
four_h = four["Elevation.3"]

one_v = savgol_filter(one_v, window_length=7, polyorder=3)
two_v = savgol_filter(two_v, window_length=7, polyorder=3)
three_v = savgol_filter(three_v, window_length=7, polyorder=3)
four_v = savgol_filter(four_v, window_length=7, polyorder=3)

one_dv = np.diff(one_v)
two_dv = np.diff(two_v)
three_dv = np.diff(three_v)
four_dv = np.diff(four_v)

threshold = 0.1
window = 5
datasets = [("0.1 m", one_v, one_dv, one_h),
             ("0.2 m", two_v, two_dv, two_h),
             ("0.3 m", three_v, three_dv, three_h),
             ("0.4 m", four_v, four_dv, four_h)]

for label, v, dv, h in datasets:
    for i in range(len(dv) - window):
        if np.all(np.abs(dv[i:i+window]) < threshold):
            print(f"{label}-- index: {i+1}, vel: {v[i+1]:.3f} m/s, height: {h[i+1]}")
            # v = v[:i+1]
            # h = v[:i+1]
            break

flows = [("Boundary Layer Height", bl_h),
         ("Skin Friction (Shear)", sf_shear),
         ("Skin Friction (Re)", sf_re),
         ("Skin Friction (Shear)", sf_mom)]
for l, i in flows:
    plt.plot(bl_x, i, label=l)
plt.grid()
plt.legend()
plt.title("Boundary Layer Characteristics Across Plate")
plt.ylabel("Boundary Layer Height (m)")
plt.xlabel("Station (m)")
plt.tight_layout()
plt.show()


#
# fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(16, 4))
#
# datasets = [
#     (one_v, one_h, "0.1 m"),
#     (two_v, two_h, "0.2 m"),
#     (three_v, three_h, "0.3 m"),
#     (four_v, four_h, "0.4 m")
# ]
#
# for ax, (v, h, lbl) in zip(axes, datasets):
#     ax.plot(v, h, label=lbl)
#     ax.grid(True)
#     # ax.legend(loc="upper right")
#     ax.set_title(lbl)
#     ax.set_aspect("auto")   # let matplotlib pick a good ratio
#
# # Shared labels
# fig.supxlabel("Velocity (m/s)")
# fig.supylabel("Elevation (m)")
# fig.suptitle("Velocity Gradient per Station Using Savitzky-Golay Filtering")
# plt.tight_layout(pad=2.0, w_pad=2.0)
# plt.show()
#
