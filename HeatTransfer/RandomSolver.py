import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


def eigen_solver(type, Bi, Ti, Tinf, h, k, L, x, alpha):

    Bi = h*L/k
    print(f"Biot: {Bi}")
    if 0.1 > Bi > 100:
        return 0

    Fo = alpha / L**2
    print(f"Fo: {Fo:.4f}t")

    def transcendental_eq(lmbd):
        if type == "slab":
            return lmbd*np.tan(lmbd) - Bi
        else if type == "cylinder":
            return 
            return 0

    guesses = [0.5, 3.5, 6.5, 9.5, 12.5]
    lambdas = []

    for guess in guesses:
        root = fsolve(transcendental_eq, guess)[0]
        lambdas.append(root)
    print(f"Eigen Values: {np.round(lambdas, 3)}")
    lambdas = np.array(lambdas)

    A = None
    if type == "slab":
        A = 2*np.sin(lambdas)/(lambdas+(np.sin(lambdas)*np.cos(lambdas)))
        # for lmbd in lambdas:
        #     A_n.append(2*np.sin(lmbd)/(lmbd+(np.sin(lmbd)*np.cos(lmbd))))

    print(f"A_n: {np.round(A, 3)}")

    t = np.linspace(0, 3600, 200)
    Fo = Fo*t
    theta = np.zeros_like(t)
    for i in range(len(lambdas)):
        theta += A[i] * np.exp(-lambdas[i]**2 * Fo) * np.cos(lambdas[i] * (x/L))

    T_series = theta * (Ti - Tinf) + Tinf

    plt.plot(t/60, T_series, label=f"Series Solution ({len(guesses)} terms)")
    plt.xlabel("Time [min]")
    plt.ylabel(f"Temperature at {x}")
    plt.title("Cooling Curve")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


eigen_solver(type="slab",  Bi=0.111, Ti=400, Tinf=300, h=100, k=45, alpha=1.2e-5, L=0.05, x=0.025)

