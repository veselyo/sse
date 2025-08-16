# Front crash example problem from Zimmerman & Hössle (2013), chapter 2.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from complete_algorithm import (Box, sse_search_algorithm, next_filename)
import __main__
from pathlib import Path


# Problem parameters.
m = 2000.0                              # Vehicle mass (kg).
g = 9.81                                # Gravitational acceleration (m/s²).
a_c = 32 * g                            # Deceleration of the passenger – 
                                        # Critical threshold level (m/s²).
v_0 = 15.6                              # Impact speed (m/s).
d_1c = d_2c = 0.3                       # Deformation limit for sections 1 and 2
                                        # respectively (m).
E_k = 0.5 * m * v_0**2                  # Impact energy (J).
x_0 = np.array([406e3, 406e3])          # Classical performance optimum.
design_space = Box(                     # Complete design space.
    np.array([0., 0.]),
    np.array([800e3, 800e3])
)                                   


# SSE parameters.
N1 = 100                                # Monte Carlo sample size phase 1.
N2 = 700                               # Monte Carlo sample size phase 2.
growth_rate = 0.05                      # Growth rate for Modification Step B.
confidence_lower = 0.99                # Confidence interval lower bound.
confidence_upper = 1.0                  # Confidence interval upper bound.
alpha_c = 0.001                         # (1 - alpha_c) is cricical confidence
                                        # level.
change_threshold = 0.01                 # Relative box size change threshold
                                        # for SSE phase I termination.


def f(args: np.ndarray):
    """Performance function.
       Returns the performance value given deformation forces F_1 and F_2 (N).
       Good designs satisfy f(F_1, F_2) ≤ 0.
    """
    if args is None:
        raise ValueError("args can't be none")
    if not isinstance(args, np.ndarray):
        raise TypeError("Argumsnts must be np arrays.")
    if args.shape != (2,):
        raise ValueError("f expects a vector of length 2: [F1, F2].")
    F_1, F_2 = args

    # Constraint 1: Impact energy needs to be smaller than the deformation
    # energy to be fully absorbed.
    if E_k > F_1 * d_1c + F_2 * d_2c:
        return 1.0
                           
    # Constraint 2: F_1 needs to be smaller than F_2 so that section 1 deforms
    # before section 2.
    if F_1 > F_2:
        return 1.0 
        
    # Otherwise, return performance. Good designs are ≤ 0, i.e. when the
    # maximum deceleration of the passenger cell (F_2 / m) is below the
    # critical threshold level a_c, and constraints are satisfied.
    return (F_2 / m - a_c) / a_c


def performance_criterion(f_x: float) -> bool:
    """
    Return True when the given performance value satisfies the performance
    criterion f(x) <= 0.
    """
    if f_x is None:
        raise ValueError("f_x can't be none.")
    if not isinstance(f_x, float):
        raise TypeError("Argument must be a scalar (float).")
    return f_x <= 0.0


def sse_complete():
    """
    Run complete SSE algorithm and plot results.
    """
    sse_box = sse_search_algorithm(
        f = f, 
        performance_criterion = performance_criterion,
        design_space = design_space,
        x_0 = x_0,
        N1 = N1,
        N2 = N2,
        growth_rate = growth_rate,
        confidence_lower = confidence_lower,
        confidence_upper = confidence_upper,
        alpha_c = alpha_c,
        change_threshold = change_threshold
    )

    # Plot
    region_cmap = ListedColormap(["lightgrey", "white"])
    F_vals = np.linspace(0, 800e3, 301)
    F_1_grid, F_2_grid = np.meshgrid(F_vals, F_vals)
    vf = np.vectorize(lambda f1, f2: f(np.array([f1, f2])))
    mask_grid = vf(F_1_grid, F_2_grid) <= 0
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.pcolormesh(F_1_grid, F_2_grid, mask_grid, cmap=region_cmap,
                  shading="auto")
    ax.plot(*x_0, marker="x", color="green", markersize=8, label="$x_0$")
    ax.set_xlabel(r"$F_1$ [N]")
    ax.set_ylabel(r"$F_2$ [N]")
    ax.set_title("Front-Crash Example – SSE solution")
    ax.set_aspect("equal", adjustable="box")
    good_patch = mpatches.Patch(facecolor="white", edgecolor="none",
                                label="Good designs")
    bad_patch  = mpatches.Patch(facecolor="lightgrey", edgecolor="none",
                                 label="Bad designs")
    x0_handle, = ax.plot(*x_0, marker="x", color="green",
                         markersize=8, label="$x_0$")
    box_handle = plt.Rectangle(
        sse_box.lower,
        *(sse_box.upper - sse_box.lower),
        facecolor="lightgreen", edgecolor="lightgreen", alpha=0.7,
        label="SSE solution box"
    )
    F1_lower = 291e3
    F1_upper = 516e3
    F2_lower = 516e3
    F2_upper = 628e3
    analytic_rect = plt.Rectangle(
        (F1_lower, F2_lower),
        F1_upper - F1_lower,
        F2_upper - F2_lower,
        facecolor="none",
        edgecolor="green",
        linestyle="--",
        linewidth=2,
        label="Analytical box"
    )
    ax.add_patch(box_handle)
    ax.add_patch(analytic_rect)
    ax.legend(
        handles=[bad_patch, good_patch, x0_handle, box_handle, analytic_rect],
        framealpha=0.7,
        loc="lower left"
    )

    main_file = getattr(__main__, "__file__", None)
    output_file = str(Path(main_file).parent) + "/outputs"
    fname = next_filename("run_{idx}_box.png", output_file)
    plt.savefig(fname, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sse_complete()