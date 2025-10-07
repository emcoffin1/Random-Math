import matplotlib.pyplot as plt
import pandas as pd


fast = pd.read_excel("drags2.xlsx", sheet_name=0, usecols="D:J", skiprows=20).dropna()
slowU = pd.read_excel("drags2.xlsx", sheet_name=0, usecols="P:V", skiprows=20).dropna()
slowR = pd.read_excel("drags2.xlsx", sheet_name=0, usecols="Z:AF", skiprows=20).dropna()

f_degree = fast["Degree"]
sU_degree = slowU["Degree (U)"]
sR_degree = slowR["Degree (R)"]

f_cp = fast["Pressure Coefficient (cp)"]
sU_cp = slowU["Pressure Coefficient (U)"]
sR_cp = slowR["Pressure Coefficient (R)"]

f_cp_min = min(f_cp)
print( f_cp_min)

# Create 3 subplots stacked vertically, sharing the x-axis
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(6, 8))

# Top subplot – Fast
ax1.plot(f_degree, f_cp, label="Fast")
ax1.set_title("Pressure Coefficient")
ax1.grid(True)
ax1.legend(loc="upper center")

# Middle subplot – Slow Unrefined
ax2.plot(sU_degree, sU_cp, label="Slow Unrefined")
ax2.grid(True)
ax2.legend(loc="upper center")

# Bottom subplot – Slow Refined
ax3.plot(sR_degree, sR_cp, label="Slow Refined")
ax3.grid(True)
ax3.legend(loc="upper center")

# Shared x-axis label and tidy layout
ax3.set_xlabel("Degrees")
ax2.set_ylabel("Pressure Coefficient")
plt.tight_layout()
plt.show()
