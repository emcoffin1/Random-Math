from scipy import integrate, linalg
import numpy as np
import matplotlib.pyplot as plt

# Number of particles
n = 1

# Bounding Box
resolution = 100
L = 1
H = 0.5

dx = L/resolution
dy = H/resolution

# Some constants
g = 9.81



def func(t, X):
    rx, ry, vx, vy = X
    






