import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.path import Path
from matplotlib import colors


def get_levels(min_z, max_z, d=1., l=0.5):
    levels = []
    j = 0
    while min_z < max_z:
        min_z = min_z + (d * j)
        levels.append(min_z)
        j += l
    return levels


def draw_isolines(low, up, func, d=0.05):
    x = np.arange(low[0], up[0], d)
    y = np.arange(low[1], up[1], d)

    z = np.zeros((len(x), len(y)))
    x, y = np.meshgrid(x, y)

    for i in range(len(x)):
        for j in range(len(x[i])):
            z[i][j] = func(np.array([x[i][j], y[i][j]]))

    levels = np.array(get_levels(np.min(z), np.max(z), d))

    return x, y, z, levels


def visualization(center, delta, func, points, down, high, d=0.05):
    fig, ax = plt.subplots()
    plt.legend(loc='upper left')
    x, y, z, levels = draw_isolines(down, high, func, d=d)
    plt.contour(x, y, z, levels=levels, zorder=1)
    ax.grid()

    if isinstance(points, np.ndarray):
        if len(points.shape) == 2:
            ax.scatter(points[:, 0], points[:, 1], color='orangered', marker='o', zorder=2)
        elif len(points.shape) == 3:
            for i in range(len(points)):
                if len(points[i]) != 0:
                    ax.scatter(points[i][:, 0], points[i][:, 1],
                               color=list(colors.XKCD_COLORS.keys())[i*7 + 25], marker='o', zorder=2)
        else:
            return None
    elif isinstance(points, list):
        if (len(points) > 0) and isinstance(points[0], np.ndarray):
            for i in range(len(points)):
                if len(points[i]) != 0:
                    ax.scatter(points[i][:, 0], points[i][:, 1],
                               color=list(colors.XKCD_COLORS.keys())[i*7 + 25], marker='o', zorder=2)

    ax.plot(center[0], center[1], color='dimgray', marker='x', zorder=3)
    verts = []

    if len(delta.shape) == 1:
        verts = [(center[0] - delta[0], center[1] - delta[1]),  # left, bottom
                 (center[0] - delta[0], center[1] + delta[1]),  # left, top
                 (center[0] + delta[0], center[1] + delta[1]),  # right, top
                 (center[0] + delta[0], center[1] - delta[1]),  # right, bottom
                 (center[0] - delta[0], center[1] - delta[1]), ]  # ignored
    elif len(delta.shape) == 2:
        verts = [(center[0] - delta[0][0], center[1] - delta[1][0]),  # left, bottom
                 (center[0] - delta[0][0], center[1] + delta[1][1]),  # left, top
                 (center[0] + delta[0][1], center[1] + delta[1][1]),  # right, top
                 (center[0] + delta[0][1], center[1] - delta[1][0]),  # right, bottom
                 (center[0] - delta[0][0], center[1] - delta[1][0]), ]  # ignored
    else:
        return None

    codes = [Path.MOVETO,
             Path.LINETO,
             Path.LINETO,
             Path.LINETO,
             Path.CLOSEPOLY, ]

    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor='none', lw=2, zorder=4)
    ax.add_patch(patch)

    plt.show()
