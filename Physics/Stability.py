import numpy as np
import matplotlib.pyplot as plt


def right_side(w, h):
    term1 = 4*9.81 / (3*h**2)
    term2 = h**2 + w**2
    term3 = np.sqrt(term2) - h
    return np.sqrt(term1*term2*term3)

def right_tilt(w, h, theta):
    h_i = w/2*np.sin(theta) + h/2*np.cos(theta)
    dh = 0.5*np.sqrt(w**2 + h**2) - h_i
    term1 = 9.81*dh/h_i**2 * (w**2 + h**2)
    return np.sqrt(2/3*term1)


if __name__ == '__main__':
    # meters
    w = 1
    h = 1.5
    theta = np.deg2rad(4)
    # T = right_side(w, h)
    T = right_tilt(w, h, theta)
    print(T)
    E = (T*(1-0.1))
    print(E)
    pe = np.abs(T-E)/T
    print(pe)

