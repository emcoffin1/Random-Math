import matplotlib.pyplot as plt


def mach_from_area_ratio_supersonic(eps, gamma=1.4, guess=2.0, tol=1e-12, itmax=60):
    # Newton's method on F(M)=A/A* - eps, constrained to M>1
    g = gamma
    expo = (g+1)/(2*(g-1))
    def F(M):
        term = (2/(g+1))*(1+(g-1)/2*M*M)
        return (1.0/M)*term**expo - eps
    def dF(M):
        term = (2/(g+1))*(1+(g-1)/2*M*M)
        dterm = (2/(g+1))*(g-1)*M
        return -(1/M**2)*term**expo + (1/M)*expo*term**(expo-1)*dterm

    M = max(1.0001, guess)
    for _ in range(itmax):
        f, fp = F(M), dF(M)
        M_new = M - f/fp
        if M_new <= 1.0:  # keep on supersonic branch
            M_new = 1.0001
        if abs(M_new - M) < tol:
            return M_new
        M = M_new
    return M  # last iterate

def area_ratio_from_M(M, gamma=1.4):
    g = gamma
    return (1.0/M) * ((2/(g+1)*(1+(g-1)/2*M*M))**((g+1)/(2*(g-1))))



