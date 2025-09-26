import matplotlib.pyplot as plt
import numpy as np

in_to_m = 0.0245

P_inj = 5 * 6894.76     # psi-pa
rho = 999   # kg/m3
g = 9.81    # m/s2
h_offset = 2.56 * in_to_m   # in-m
orif_area = 0.0005245 * 0.092903    # ft2
h_init = 5.70866 * in_to_m
h_init_act = h_init - h_offset

h_final = 6.9685 * in_to_m
h_final_act = h_final - h_offset

t = 51.04
dm = 9.6 * 0.453592
dmdt = dm/t

h_avg = (h_init_act + h_final_act) / 2  # m2
P_back = rho * g * h_avg

cd = dmdt / (orif_area * np.sqrt(2*rho*(P_inj-P_back)))

print(f"{cd:.3f}")




