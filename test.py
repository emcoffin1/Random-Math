import numpy as np

R_UNIV = 8.314462618  # J/mol-K

class Species:
    """NASA 7/9-coefficient polynomial for one species"""
    def __init__(self, name, coeffs_low, coeffs_high, t_low, t_high, t_mid, MW):
        self.name = name
        self.coeffs = {'low': coeffs_low, 'high': coeffs_high}
        self.t_low, self.t_high, self.t_mid = t_low, t_high, t_mid
        self.MW = MW  # kg/mol

    def _region(self, T):
        return 'low' if T <= self.t_mid else 'high'

    # -------- thermodynamic functions (molar) -------- #
    def cp_mol(self, T):
        a = self.coeffs[self._region(T)]
        return R_UNIV * (a[0] + a[1]*T + a[2]*T**2 + a[3]*T**3 + a[4]*T**4)

    def h_mol(self, T):
        a = self.coeffs[self._region(T)]
        return R_UNIV * T * (a[0] + a[1]*T/2 + a[2]*T**2/3 +
                             a[3]*T**3/4 + a[4]*T**4/5 + a[5]/T)

    def s_mol(self, T):
        a = self.coeffs[self._region(T)]
        return R_UNIV * (a[0]*np.log(T) + a[1]*T +
                         a[2]*T**2/2 + a[3]*T**3/3 + a[4]*T**4/4 + a[6])

    # -------- convenience (mass basis) -------- #
    def cp_mass(self, T):
        return self.cp_mol(T) / self.MW

    def h_mass(self, T):
        return self.h_mol(T) / self.MW

    def s_mass(self, T):
        return self.s_mol(T) / self.MW


def read_nasa9(filepath, wanted):
    species_data = {}
    with open(filepath) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        name = line[0:24].strip()
        if name in wanted:
            t_low, t_high, t_mid = map(float, line[45:73].split())
            coeffs_high = [float(x.replace('D','E')) for x in ''.join(lines[i+1:i+3]).split()]
            coeffs_low  = [float(x.replace('D','E')) for x in ''.join(lines[i+3:i+5]).split()]
            MW = float(line[52:65]) * 1e-3 if line[52:65].strip() else 0.0
            species_data[name] = Species(name, coeffs_low, coeffs_high, t_low, t_high, t_mid, MW)
    return species_data


l = read_nasa9("nasa9.dat.txt", ["CO2", "H2O", "CO", "H2", "O2", "N2", "OH", "O", "H"])
print(l)