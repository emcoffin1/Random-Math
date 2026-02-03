import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import solve_ivp

g = 9.81

# ---------- Engine model ----------
# T_MAX = 3682.0  # N (3.682 kN)
T_MIN = 423

def mdot(throttle):
    """kg/s"""
    return max(0.0, 1.5 * throttle - 5.92e-4)


def throttle_from_thrust(T, T_MAX, u_min, u_max):
    """Throttle in [0,1] from thrust in N"""
    T = np.clip(T, 0.0, T_MAX)
    u = 0.259 * (T / 1000) + 0.106
    return np.clip(u, u_min, u_max)


# ---------- Controller ----------
def acceleration_cmd(z, v, z_ref, Kp, Kd, a_max):
    e = z_ref - z
    edot = -v
    return np.clip(Kp * e + Kd * edot, -g, g*a_max)


# ---------- Thrust --------------
def thrust_from_throttle(u, T_MAX):
    T = 1000 * (u - 0.106) / 0.259
    return np.clip(T, T_MIN, T_MAX)


# ---------- Simulation ----------

def simulate(m0, z_ref, Kp, Kd, a_max, t_hover, u_min, u_max, dt=0.001, t_final=10.0):
    T_MAX = thrust_from_throttle(max_u, 3682)
    N = int(t_final / dt)

    state = 0

    z = 0.0
    v = 0.0
    m = m0
    t = 0

    t_hist, z_hist, v_hist, m_hist, u_hist, T_hist, a_hist = [], [], [], [], [], [], []

    for k in range(N):
        t += dt

        # --- Controller ---
        a_cmd = acceleration_cmd(z, v, z_ref, Kp, Kd, a_max)

        T_cmd = m * (g + a_cmd)

        u = throttle_from_thrust(T_cmd, T_MAX, u_min, u_max)
        T = thrust_from_throttle(u, T_MAX)

        # --- Dynamics ---
        a = (T / m) - g
        v += a * dt
        z += v * dt
        m -= mdot(u) * dt

        # --- Save ---
        t_hist.append(t)
        z_hist.append(z)
        v_hist.append(v)
        m_hist.append(m)
        u_hist.append(u)
        T_hist.append(T)
        a_hist.append(a/g)

        if z < 0:
            z = 0
            v = 0

        if state == 0:

            err = np.abs(z - z_ref) / z
            if err < 0.01:
                print(f"50m reached: {t:.2f}s")
                print(f"Fuel consumed: {(m0 - m):.2f} kg ~~ {((m0-m) * 2.20462):.2f}lbs")
                t_stop = t+t_hover
                state = 1
                # break

        elif state == 1:
            err = np.abs(t - t_stop) / t_stop
            if err < 0.01:
                print(f"Hover Completed: {t:.2f}s")
                print(f"Fuel consumed: {(m0 - m):.2f} kg ~~ {((m0 - m) * 2.20462):.2f}lbs")
                # break
                z_ref = 0.05
                state = 2

        elif state == 2:
            err = z - z_ref
            if err < 0.01:
                print(f"0m reached: {t:.2f}s")
                print(f"Fuel consumed: {(m0 - m):.2f} kg ~~ {((m0 - m) * 2.20462):.2f}lbs")
                break

    return (
        np.array(t_hist),
        np.array(z_hist),
        np.array(v_hist),
        np.array(m_hist),
        np.array(u_hist),
        np.array(T_hist),
        np.array(a_hist),
    )


def derivatives(state, z_ref, Kp, Kd, a_max, T_MAX, u_min, u_max):
    z, v, m = state

    a_cmd = acceleration_cmd(z,v,z_ref,Kp,Kd, u_max)
    T_cmd = m * (g + a_cmd)

    u = throttle_from_thrust(T_cmd, T_MAX, u_min, u_max)
    T = thrust_from_throttle(u, T_MAX)
    a = T/m - g
    mdot_val = mdot(u)

    dxdt = v
    dvdt = a
    dmdt = -mdot_val

    return np.array([dxdt, dvdt, dmdt]), u ,T, a

def simulate_rk4(m0, z_ref, Kp, Kd, a_max, t_hover,
                 u_min, u_max, dt=0.001, t_final=10.0):

    T_MAX = thrust_from_throttle(u_max, 4000)
    N = int(t_final / dt)
    task = 0

    # Initial state
    state = np.array([0.0, 0.0, m0])
    t = 0.0

    # Logs
    t_hist, z_hist, v_hist, m_hist = [], [], [], []
    u_hist, T_hist, a_hist = [], [], []

    for _ in range(N):
        t += dt

        # --- RK4 ---
        k1, u1, T1, a1 = derivatives(state, z_ref, Kp, Kd, a_max, T_MAX, u_min, u_max)
        k2, _,  _,  _  = derivatives(state + 0.5*dt*k1, z_ref, Kp, Kd, a_max, T_MAX, u_min, u_max)
        k3, _,  _,  _  = derivatives(state + 0.5*dt*k2, z_ref, Kp, Kd, a_max, T_MAX, u_min, u_max)
        k4, _,  _,  _  = derivatives(state + dt*k3,     z_ref, Kp, Kd, a_max, T_MAX, u_min, u_max)

        state = state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

        z, v, m = state

        # Ground contact
        if z < 0:
            z = 0
            v = 0
            state[0] = 0
            state[1] = 0

        # --- Log (use k1 values) ---
        t_hist.append(t)
        z_hist.append(z)
        v_hist.append(v)
        m_hist.append(m)
        u_hist.append(u1)
        T_hist.append(T1)
        a_hist.append(a1 / g + 1)  # felt g-load

        if task == 0:
            if abs(z - z_ref) <= 0.01:
                print(f"Ascent Completed to {z_ref:.2f}m in {t:.2f}s")
                fc = m_hist[0] - m_hist[-1]
                print(f"Fuel Consumed: {fc:.2f} kg ~~ {fc*2.20462:.2f}lbs")
                task += 1
                t_start = t + t_hover

        elif task == 1:
            if abs(t-t_start) <= 0.01:
                print(f"Hover Completed for {t_hover:.2f}s")
                fc = m_hist[0] - m_hist[-1]
                print(f"Fuel Consumed: {fc:.2f} kg ~~ {fc * 2.20462:.2f}lbs")
                task += 1
                z_ref = 0


        elif task == 2:
            if abs(z - z_ref) <= 0.01:
                print(f"Descent Completed to {z_ref:.2f}m in {t:.2f}s")
                fc = m_hist[0] - m_hist[-1]
                print(f"Fuel Consumed: {fc:.2f} kg ~~ {fc*2.20462:.2f}lbs")
                break

    return (
        np.array(t_hist),
        np.array(z_hist),
        np.array(v_hist),
        np.array(m_hist),
        np.array(u_hist),
        np.array(T_hist),
        np.array(a_hist),
    )


def ode(t, state, z_ref, Kp, Kd, a_max, T_MAX, u_min, u_max):
    z, v, m = state
    a_cmd = acceleration_cmd(z,v,z_ref,Kp,Kd, u_max)
    T_cmd = m * (g + a_cmd)
    u = throttle_from_thrust(T_cmd, T_MAX, u_min, u_max)
    T = thrust_from_throttle(u, T_MAX)
    a = T/m-g
    mdot_val = mdot(u)
    return [v, a, -mdot_val]

def simulate_variable(state, z_ref, Kp, Kd, a_max, T_max, u_min, u_max, m0):
    sol = solve_ivp(ode, t_span=(0.0, 100), y0=[0.0, 0.0, state[2]], args=(z_ref, Kp, Kd, a_max, T_MAX, u_min, u_max),
                    method = "DP853", rtol=1e-8,atol=1e-10, max_step=0.05)


# ---------- Run ----------
if __name__ == "__main__":
    m0 = 158     # kg
    z_ref = 50.0   # m
    t_final = 100
    mdot_calc_0 = 0
    iter = 50
    a_max = 2
    t_hover = 0.5
    max_u = 1.0
    min_u = 0.15
    T_MAX = thrust_from_throttle(max_u, 4000)

    # Gains (start here)
    Kp = 1.2
    Kd = 2.0

    # t, z, v, m, u, T, a = simulate(m0, z_ref, Kp, Kd, a_max=a_max, t_hover=t_hover,
    #                                dt=0.001, t_final=t_final, u_min=min_u, u_max=max_u)



    t, z, v, m, u, T, a = simulate_rk4(
        m0, z_ref, Kp, Kd,
        a_max=a_max,
        t_hover=t_hover,
        dt=0.001,
        t_final=t_final,
        u_min=min_u,
        u_max=max_u
    )

    print(f"Minimum throttle achieved: {np.min(u):.2f}")
    print(f"Maximum g-load achieved: {np.max(a):.2f}")

    pd = pd.DataFrame({
        "t": t,
        "z": z,
        "v": v,
        "a": a,
        "u": u,
    })


    # pd.to_csv("thrust_optimization.csv", index=False)

    plt.figure(figsize=(10,6))
    plt.subplot(5,1,1)
    plt.plot(t, z)
    plt.ylabel("Altitude (m)")
    plt.grid()

    plt.subplot(5,1,2)
    plt.plot(t, v)
    plt.ylabel("Velocity (m/s)")
    plt.grid()

    plt.subplot(5,1,3)
    plt.plot(t, m)
    plt.ylabel("Mass (kg)")
    plt.xlabel("Time (s)")
    plt.grid()

    plt.subplot(5,1,4)
    plt.plot(t, u)
    plt.ylabel("Throttle (%)")
    plt.xlabel("Time (s)")
    plt.grid()

    plt.subplot(5,1,5)
    plt.plot(t, a)
    plt.ylabel("g-Load (g's)")
    plt.xlabel("Time (s)")
    plt.grid()

    plt.tight_layout()
    plt.show()
