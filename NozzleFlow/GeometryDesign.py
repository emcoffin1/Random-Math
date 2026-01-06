import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.path import Path
from matplotlib.patches import PathPatch


def view_channel_slices(data: dict):
    N = len(data["E"]["x"])
    N_ch = int(data["C"]["num_ch"])
    fin_t = data["C"]["spacing"]
    chan_width = data["C"]["width_arr"]
    chan_depth = data["C"]["depth_arr"]
    r_gas = data['E']["y"]
    t_wall = data["W"]["thickness"]

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    rc_outer_const = np.max(r_gas) + t_wall + 1.2* np.max(chan_depth)


    def channel_plot(ax, r_wall, w, d, theta0=0.0):
        # Single Channel
        n = np.array([np.cos(theta0), np.sin(theta0)])
        t = np.array([-np.sin(theta0), np.cos(theta0)])
        c = r_wall * n

        corners = np.array([
            c + (-w / 2) * t + 0 * n,
            c + (w / 2) * t + 0 * n,
            c + (w / 2) * t + d * n,
            c + (-w / 2) * t + d * n,
        ])

        ax.fill(corners[:, 0], corners[:, 1], color="r", alpha=0.7)

    def draw_slice(i):
        ax.clear()

        rg = r_gas[i]
        rw = rg + t_wall
        # rw = rg + 0.015
        w = chan_width[i]
        d = chan_depth[i]
        f = fin_t

        theta = np.linspace(0, 2*np.pi, 500)

        # Gas wall
        # ax.add_patch(plt.Circle((0,0), rg, fill=False, linestyle="--", edgecolor="k"))

        # Gas and thick wall
        xi = rg * np.cos(theta[::-1])
        yi = rg * np.sin(theta[::-1])

        xo = rw * np.cos(theta)
        yo = rw * np.sin(theta)

        # Inner wall and outer wall
        ax.fill(np.concatenate([xi, xo]), np.concatenate([yi, yo]), color="k", alpha=0.7)


        for n in range(N_ch):
            theta0 = n *2 * np.pi / N_ch
            channel_plot(ax, r_wall=rw, w=w, d=d, theta0=theta0)


        # Channel
        # ax.add_patch(plt.Rectangle((-w/2, rw), w, chan_depth[i], color="tab:blue", alpha=0.7))
        #
        # # Fin
        # ax.add_patch(plt.Rectangle((w/2, rw), f, chan_depth[i], color="tab:orange", alpha=0.6))

        lim = rc_outer_const * 1.2
        # r_max = np.max(r_)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Axial Station {i}")
        ax.axis('off')


    draw_slice(0)

    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, "Station", 0, N - 1, valinit=0, valstep=1)

    def update(val):
        draw_slice(int(slider.val))
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()
