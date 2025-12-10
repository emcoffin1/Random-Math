"""
https://ntrs.nasa.gov/api/citations/20020085330/downloads/20020085330.pdf
https://shepherd.caltech.edu/EDL/PublicResources/sdt/SDToolbox/cti/NASA9/nasa9.dat
Species: CO2 H2O CO H2 O2 N2 OH O H
CO2: 2586:2597  -->
H2O: 5362:5370  -->
CO:  2500:2511  -->
H2:  5300:5311  -->
O2:  7366:7377  -->
N2:  6800:6811  -->
OH:  7336:7347  -->
O:   7287:7298  -->
H:   4994:5005  -->
"""
import numpy as np

class Species:
    def __init__(self, name, temp_cutoff, activation, coeffs_low, coeffs_high, int_const_low, int_const_high,
                 mw, element_dict):
        self.R = 8.314462618  # J/mol-K

        self.name = name
        self.activation = float(activation.strip().replace('D', 'E'))
        self.temp_cutoff = self.chunk(temp_cutoff, n=11)[1]
        self.coeffs = {"low": self.chunk(coeffs_low), "high": self.chunk(coeffs_high)}
        self.int_const = {"low": self.chunk(int_const_low), "high": self.chunk(int_const_high)}
        self.mw = float(mw) / 1000.0
        self.element_dict = element_dict
        self.elem_vec = np.array([element_dict.get(e, 0) for e in ['C', 'H', 'O', "N"]])  # C, H, O

    def chunk(self, coeffs, n=16) -> list:
        # Split string into n-character chunks
        x = [coeffs[i:i + n] for i in range(0, len(coeffs), n)]
        # Clean each chunk: strip whitespace and convert Fortran D to E
        x = [float(chunk.strip().replace('D', 'E')) for chunk in x if chunk.strip()]
        return x

    def _range(self, T):
        return "low" if T < self.temp_cutoff else "high"

    # -- Thermo functions (molar basis) -- #

    def cp_mol(self, T):
        a = self.coeffs[self._range(T)]
        cp = a[0]*T**-2 + a[1]*T**-1 + a[2] + a[3]*T + a[4]*T**2 + a[5]*T**3 + a[6]*T**4
        return self.R * cp  # J/mol-K

    def h_mol(self, T):
        a = self.coeffs[self._range(T)]
        b = self.int_const[self._range(T)]
        h = -a[0]*T**-2 + a[1]*np.log(T)/T + a[2] + a[3]*T/2 + a[4]*T**2/3 + a[5]*T**3/4 + a[6]*T**4/5 + b[0]/T
        return self.R * h * T

    def s_mol(self, T):
        a = self.coeffs[self._range(T)]
        b = self.int_const[self._range(T)]
        s = -a[0]*T**-2/2 - a[1]/T + a[2]*np.log(T) + a[3]*T + a[4]*T**2/2 + a[5]*T**3/3 + a[6]*T**4/4 + b[1]
        return self.R * s

    def g_mol(self, T):
        return self.h_mol(T) - T * self.s_mol(T)

    # -- MASS BASIS -- #
    def cp_mass(self, T):
        return self.cp_mol(T) / self.mw

    def h_mass(self, T):
        return self.h_mol(T) / self.mw

    def s_mass(self, T):
        return self.s_mol(T) / self.mw

    def g_mass(self, T):
        return self.g_mol(T) / self.mw


def get_species_data():
    spec_dict = {}

    with open("nasa9.dat.txt", "r") as f:
        lines = f.readlines()
    temprow1 = 2
    temprow2 = 5
    # tempcol = 0:22
    # actcol = 65:80
    # coeffcol = 0:80
    # coeffcol2 = 0:32
    # intconstcol = 48:80


    CO2 = lines[2585:2598]
    H2O = lines[5361:5369]
    CO = lines[2499:2510]
    H2 = lines[5299:5310]
    O2 = lines[7365:7376]
    N2 = lines[6799:6810]
    OH = lines[7335:7346]
    O = lines[7286:7297]
    H = lines[4993:5004]

    spec_list = {"CO2":CO2, "H2O":H2O, "CO":CO, "H2":H2, "O2":O2, "N2":N2, "OH":OH, "O":O, "H":H}
    # spec_list = {"CO2": CO2, "H2O": H2O, "CO": CO, "H2": H2, "O2": O2, "OH": OH, "O": O, "H": H}
    elements = ['C', 'H', 'O', "N"]
    elem_matrix = np.array([
        [1, 0, 1, 0, 0, 0, 0, 0, 0],     # C
        [0, 2, 0, 2, 0, 0, 1, 0, 1],     # H
        [2, 1, 1, 0, 2, 0, 1, 1, 0],      # O
        [0, 0, 0, 0, 0, 2, 0, 0, 0]      # N
    ])

    for i, (k,v) in enumerate(spec_list.items()):
        element_dict = {elem: elem_matrix[j, i] for j, elem in enumerate(elements) if elem_matrix[j, i] != 0}
        spec_dict[k] = Species(name=k, temp_cutoff=v[temprow1][0:22], activation=v[temprow1][65:80],
                               coeffs_low=v[temprow1+1][0:80]+v[temprow1+2][0:32],
                               coeffs_high=v[temprow2+1][0:80]+v[temprow2+2][0:32],
                               int_const_low=v[temprow1+2][48:80],
                               int_const_high=v[temprow2+2][48:80],
                               mw=v[temprow1-1][52:65].strip(),
                               element_dict=element_dict)

    return spec_dict, elements


if __name__ == '__main__':

    # spec: dict[species]
    # elem: list of elements
    spec, elem = get_species_data()

