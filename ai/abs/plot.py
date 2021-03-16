from abs import deeppoly

import matplotlib.pyplot as plt
import numpy as np


def plot_deeppoly(ele: deeppoly.Ele,
                  x_lim=(-1, 1),
                  y_lim=(-1, 1),
                  fig: plt.Figure = None):

    lb = ele.lb().detach().numpy().squeeze()
    ub = ele.ub().detach().numpy().squeeze()
    step_size = 0.01

    assert len(lb) == 2

    h_bound_x = np.arange(lb[0], ub[0], step_size)
    h_bound_y1 = np.full_like(h_bound_x, lb[1])
    h_bound_y2 = np.full_like(h_bound_x, ub[1])

    v_bound_y = np.arange(lb[1], ub[1], step_size)
    v_bound_x1 = np.full_like(v_bound_y, lb[0])
    v_bound_x2 = np.full_like(v_bound_y, ub[0])

    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.get_axes()[0]

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.grid(True)

    ax.plot(h_bound_x, h_bound_y1, "r-", alpha=0.5)
    ax.plot(h_bound_x, h_bound_y2, "r-", alpha=0.5)
    ax.plot(v_bound_x1, v_bound_y, "r-", alpha=0.5)
    ax.plot(v_bound_x2, v_bound_y, "r-", alpha=0.5)
    ax.fill_between(h_bound_x, h_bound_y1, h_bound_y2, alpha=0.5)

    return fig, ax
