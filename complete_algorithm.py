# Full two-phase SSE search algorithm from Zimmerman & Hössle (2013).

from dataclasses import dataclass
import numpy as np
from scipy.stats import beta
from typing import Callable, Tuple, List
from time import perf_counter
from pathlib import Path
import re
import __main__
import pandas as pd
import matplotlib.pyplot as plt


def next_filename(file_name: str, directory: str) -> str:
    """
    Return file_name containing the {idx} placeholder, replacing it by the
    smallest positive integer starting at 1 that does not yet exist in
    directory.

    Parameters:
        file_name: str => Must contain the literal substring "{idx}".
            "run_{idx}.png" is an example.
            
        directory: str => folder to scan
    """
    # Type check.
    if not isinstance(file_name, str):
        raise TypeError("File name must be str.")
    if not isinstance(directory, str):
        raise TypeError("Directory must be str.")
    
    # Contains {idx} check.
    if "{idx}" not in file_name:
        raise ValueError("File name must contain the '{idx}' placeholder.")

    # Make directory if doesn't exist.
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)

    # Build a regex that matches existing files with digits in place of {idx}.
    regex = re.escape(file_name).replace(r"\{idx\}", r"(\d+)")
    rx = re.compile(rf"^{regex}$")

    indices = []
    for f in dir_path.iterdir():
        m = rx.match(f.name)
        if m:
            indices.append(int(m.group(1)))

    next_idx = max(indices) + 1 if indices else 1
    filename = file_name.replace("{idx}", str(next_idx))
    return str(dir_path / filename)


@dataclass
class Box:
    # Bounds are of shape (p,) where p is the number of dimensions. They are
    # treated as open bounds, i.e. a point equal to their value is considered to
    # lie outside the box.
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self):        
        # None check.
        if self.lower is None or self.upper is None:
            raise ValueError("Boundaries can't be none.")
        
        # Type check.
        if not isinstance(self.lower, np.ndarray):
            raise TypeError("Boundaries must be np arrays.")
        if not isinstance(self.upper, np.ndarray):
            raise TypeError("Boundaries must be np arrays.")
        
        # Shape check.
        if self.lower.ndim != 1 or self.upper.ndim != 1:
            raise ValueError("Box bounds must be 1-dimensional arrays.")
        if self.lower.shape != self.upper.shape:
            raise ValueError("Lower and upper boundary shape must be equal")
        
        # Size check.
        if self.lower.size == 0 or self.upper.size == 0:
            raise ValueError("Boundaries can't be of size 0.")

        # Value type check.
        if not isinstance(self.lower[0], float):
            raise TypeError("Boundaries values must be floats.")
        if not isinstance(self.upper[0], float):
            raise TypeError("Boundaries values must be floats.")
        
        # Valid bounds check.
        if not np.all(self.lower <= self.upper):
            raise ValueError("All lower boundaries must be less or equal to" \
                             "corresponding upper boundaries.")
        
        # Finite check.
        if np.isinf(self.lower).any() or np.isinf(self.upper).any():
            raise ValueError("Box bounds must be finite.")

    # Return size of this box.
    def size(self) -> float:
        return float(np.prod(self.upper - self.lower))

    # Return a copy of this box.
    def copy(self):
        return Box(self.lower.copy(), self.upper.copy())

    # Given N samples, return an array of shape (N,) indicating whether all
    # samples are within the boundaries of this box.
    def contains(self, samples: np.ndarray) -> np.ndarray:
        # None check.
        if samples is None:
            raise ValueError("Samples can't be none.")

        # Type check.
        if not isinstance(samples, np.ndarray):
            raise TypeError("Samples must be np arrays.")
        
        # Shape check.
        if samples.ndim != 2 or samples.shape[1] != self.lower.shape[0]:
            raise ValueError("Samples must be a 2-D array of shape (N, p)" \
                             "with the same p as the box dimension.")

        # Size check.
        if samples.size == 0:
            raise ValueError("Samples can't be of size 0.")
        
        # Value type check.
        if not isinstance(samples[0][0], float):
            raise TypeError("Sample values must be floats.")
        
        return np.all((samples > self.lower) & (samples < self.upper), axis=1)


def modification_step_A(given_box: Box,
                        good_designs: np.ndarray,
                        bad_designs: np.ndarray) -> Box:
    """
    Performs Modification Step A on a given box with N Monte Carlo samples.
    
    Parameters:
        given_box: shape (p,) => Given box.
        good_designs: shape (m, p) => Samples inside the box that satisfy the
            performance criterion.
        bad_designs: shape ((N - m), p) => Samples that do not satisfy the
            performance criterion. They must be sorted by their performance
            value in descending order.

    Returns:
        A box with the largest volume that contains no bad designs of the
        given Monte Carlo sample.
    """
    # None checks.
    if given_box is None:
        raise ValueError("Given box can't be none.")
    if good_designs is None:
        raise ValueError("Good designs can't be none.")
    if bad_designs is None:
        return given_box

    # Type checks.
    if not isinstance(given_box, Box):
        raise TypeError("Given box must be of Box type.")
    if not isinstance(good_designs, np.ndarray):
        raise TypeError("Good designs must be np arrays.")
    if not isinstance(bad_designs, np.ndarray):
        raise TypeError("Bad designs must be np arrays.")

    # Shape checks.
    p = good_designs.shape[1]
    if good_designs.ndim != 2:
        raise ValueError("Good designs must be 2D arrays.")
    if bad_designs.shape[1] != p or bad_designs.ndim != 2:
        raise ValueError("Good and bad design dimension must be equal.")
    if given_box.lower.shape[0] != p:
        raise ValueError("Given box and sample dimensions must be equal.")

    # Size checks.
    if given_box.size() == 0:
        raise ValueError("Can't perform step A on box of size 0.")
    if bad_designs.size == 0:
        return given_box
    if good_designs.size == 0:
        raise ValueError("There must be at least one good design.")

    # Value type checks.
    if not isinstance(bad_designs[0][0], float):
        raise TypeError("Bad designs must be np arrays of floats.")
    if not isinstance(good_designs[0][0], float):
        raise TypeError("Good designs must be np arrays of floats.")
    
    # Designs inside box check.
    if not np.all(given_box.contains(good_designs)):
        raise ValueError("All good designs must lie inside the given box.")
    if not np.all(given_box.contains(bad_designs)):
        raise ValueError("All bad designs must lie inside the given box.")

    # Keep track of the best encountered box.
    largest_box = None
    largest_size = -np.inf

    # For each good design:
    for A in good_designs:
        # Make a copy of the original box.
        box_A = given_box.copy()

        # For each bad design:
        for B in bad_designs:
            lost_good_designs = []
            prev_good_designs = box_A.contains(good_designs)
            
            # For each dimension (index i):
            for i in range(p):
                # Create a temporary box for evaluating number of lost good
                # designs in dimension i.
                box_A_i = box_A.copy()

                # Compare positions of designs A and B in dimension i and 
                # relocate upper or lower boundary to exclude bad design B.
                if A[i] < B[i]:
                    box_A_i.upper[i] = B[i]
                else:
                    box_A_i.lower[i] = B[i]

                # Remember number of lost good designs.
                lost_good_designs.append(
                    np.sum(prev_good_designs & ~box_A_i.contains(good_designs))
                )

            # Choose an dimension with the minimal loss of good designs.
            min_loss_dim = np.argmin(lost_good_designs)

            # Compare positions of designs A and B in this dimenision and 
            # shrink upper or lower boundary to exclude bad design B.
            if A[min_loss_dim] < B[min_loss_dim]:
                box_A.upper[min_loss_dim] = min(B[min_loss_dim],
                                                box_A.upper[min_loss_dim])
            else:
                box_A.lower[min_loss_dim] = max(box_A.lower[min_loss_dim],
                                                B[min_loss_dim])

        # Update max volume and best box if appropriate.
        size_A = box_A.size()
        if largest_size < size_A:
            largest_size = size_A
            largest_box = box_A.copy()

    return largest_box


def modification_step_B(
    given_box: Box,
    design_space: Box,
    growth_rate: float) -> Box:
    """
    Perform Modification Step B on a given box.

    Parameters:
        given_box: shape (p,) => Given box.
        design_space: shape (p,) => Set of all possible designs.
        growth_rate: constant => Growth rate used to extend the box.

    Assumptions:
        Every dimension is considered equally likely to provide more good input
        space.
        
    Returns:
        A box with expanded boundaries.
    """
    # None checks.
    if given_box is None:
        raise ValueError("Given box can't be none.")
    if design_space is None:
        raise ValueError("Design space can't be none.")
    if growth_rate is None:
        raise ValueError("Growth rate can't be none.")
    
    # Type checks.
    if not isinstance(given_box, Box):
        raise TypeError("Given box must be of Box type.")
    if not isinstance(design_space, Box):
        raise TypeError("Design space must be of Box type.")
    if not isinstance(growth_rate, float):
        raise TypeError("Growth rate must be a float.")

    # Shape check.
    p = design_space.lower.shape[0]
    if given_box.lower.shape[0] != p:
        raise ValueError("Dimension of box must be equal to the design space.")

    # Value check.
    if growth_rate < 0:
        raise ValueError("Growth rate must be >= 0.")
    if not np.all((given_box.lower >= design_space.lower) &
                  (given_box.upper <= design_space.upper)):
        raise ValueError("Box must lie inside the design space.")
        
    # Perform extension across every dimension.
    enlarged_box = given_box.copy()
    extension_sizes = growth_rate * (design_space.upper - design_space.lower)
    enlarged_box.lower = np.maximum(design_space.lower,
                                    (enlarged_box.lower - extension_sizes))
    enlarged_box.upper = np.minimum(design_space.upper,
                                    (enlarged_box.upper + extension_sizes))
    return enlarged_box


def monte_carlo_sample(N: int,
                       given_box: Box,
                       f: Callable[[np.ndarray], float],
                       performance_criterion: Callable[[float], bool]
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a box, draws N Monte Carlo samples inside it. Then, given a
    performance criterion, seperate the samples into good designs that satisfy
    it, and bad designs sorted by their performance value in descending order.

    Parameters:
        N: constant => Number of Monte Carlo samples.
        given_box: shape (p,) => Box in which Monte Carlo sample is computed.
        f: function(shape (p,)) -> float => Performance function.
        performance_criterion: function(float) -> bool => Given a perfomance
            value, return whether it satisfies the performance criterion or not.
        
    Returns:
        Tuple of good designs with shape (m, p) and sorted bad designs of shape
        (N-m, p).
    """
    # None checks.
    if N is None:
        raise ValueError("N can't be none.")
    if given_box is None:
        raise ValueError("Given box can't be none.")
    if f is None:
        raise ValueError("Performance function can't be none.")
    if performance_criterion is None:
        raise ValueError("Performance criterion can't be none.")

    # Type checks
    if not isinstance(N, int):
        raise TypeError("N must be int.")
    if not isinstance(given_box, Box):
        raise TypeError("Given box must be of Box type.")
    
    p = given_box.lower.shape[0]

    try:
        test_val = f(np.zeros(p))
    except Exception:
        raise TypeError("Performance function f must accept a (p,) array.")
    if not isinstance(test_val, float):
        raise TypeError("Performance function f must return a float.")
    
    try:
        test_val = performance_criterion(float(test_val))
    except Exception:
        raise TypeError("Performance_criterion must accept a float.")
    if not isinstance(test_val, bool):
        raise TypeError("Performance_criterion must return bool.")

    # Size check.
    if given_box.size() == 0:
        raise ValueError("Can't sample in a box with size 0.")

    # Value check.
    if N <= 0:
        raise ValueError("N must be positive.")

    # Draw N Monte Carlo samples inside the box
    samples = np.random.uniform(given_box.lower, given_box.upper, size=(N, p))

    # Compute performance value for every sample
    perfomance_vals = np.apply_along_axis(f, 1, samples)

    # Split into good and bad designs
    good_mask = np.vectorize(performance_criterion)(perfomance_vals)
    good_designs = samples[good_mask]
    bad_designs = samples[~good_mask]

    # Sort bad designs if there are any
    if bad_designs.size != 0:
        bad_designs_desc_order = np.argsort(perfomance_vals[~good_mask])[::-1]
        bad_designs = bad_designs[bad_designs_desc_order]

    return good_designs, bad_designs


def sse_search_algorithm(
        f: Callable[[np.ndarray, float], float],
        performance_criterion: Callable[[float], bool],
        design_space: Box,
        x_0: np.ndarray,
        N1: int,
        N2: int,
        growth_rate: float,
        confidence_lower: float,
        confidence_upper: float,
        alpha_c: float,
        change_threshold: float,
        params_names: List[str]) -> Box:
    """
    Run the two-phase SSE search algorithm.

    Parameters:
        f: function(shape (p,), target) -> float => Performance function.
        performance_criterion: function(float) -> bool => Given a perfomance
            value, return whether it satisfies the performance criterion or not.
        design_space: shape (p,) => Set of all possible designs.
        x_0: shape (p,) => Good design found by classical optimization on the
            design space.
        N1: constant => Monte-Carlo sample size for phase 1.
        N2: constant => Monte-Carlo sample size for phase 2.
        growth_rate: constant => Used for Modification Step B box expansion.
        confidence_lower: constant => Confidence interval lower bound. 
        confidence_upper: constant => Confidence interval upper bound.
        alpha_c: constant => (1 - alpha_c) is the confidence level.
        change_threshold: constant => Relative box size change threshold
            for phase I termination.
        params_names: List[str] => Names for parameters shown in result plot.

    Returns:
        Final solution box.
    """
    start_time = perf_counter()

    # None checks.
    if design_space is None:
        raise ValueError("Design space can't be none.")
    if f is None:
        raise ValueError("Performance function can't be none.")
    if performance_criterion is None:
        raise ValueError("Performance criterion can't be none.")
    if x_0 is None:
        raise ValueError("x_0 can't be none.")
    if N1 is None:
        raise ValueError("N1 can't be none.")
    if N2 is None:
        raise ValueError("N2 can't be none.")
    if growth_rate is None:
        raise ValueError("Growth rate can't be none.")
    if confidence_lower is None:
        raise ValueError("Confidence interval lower can't be none.")
    if confidence_upper is None:
        raise ValueError("Confidence interval upper can't be none.")
    if alpha_c is None:
        raise ValueError("alpha_c can't be none.")
    if change_threshold is None:
        raise ValueError("Change threshold can't be none.")
    if params_names is None:
        raise ValueError("params_names can't be none.")
    
    # Type checks
    if not isinstance(design_space, Box):
        raise TypeError("Design space must be of Box type.")

    p = design_space.lower.shape[0]

    try:
        test_val = f(np.zeros(p))
    except Exception:
        raise TypeError("Performance function f must accept a (p,) array.")
    if not isinstance(test_val, float):
        raise TypeError("Performance function f must return a float.")
    
    try:
        test_val = performance_criterion(float(test_val))
    except Exception:
        raise TypeError("Performance_criterion must accept a float.")
    if not isinstance(test_val, bool):
        raise TypeError("Performance_criterion must return bool.")
    
    if not isinstance(x_0, np.ndarray):
        raise TypeError("x_0 must be an np array.")
    if not isinstance(N1, int):
        raise TypeError("N1 must be int.")
    if not isinstance(N2, int):
        raise TypeError("N2 must be int.")
    if not isinstance(growth_rate, float):
        raise TypeError("Growth rate must be a float.")
    if not isinstance(confidence_lower, float):
        raise TypeError("Confidence interval lower must be a float.")
    if not isinstance(confidence_upper, float):
        raise TypeError("Confidence interval upper must be a float.")
    if not isinstance(alpha_c, float):
        raise TypeError("alpha_c must be a float.")
    if not isinstance(change_threshold, float):
        raise TypeError("Change threshold must be a float.")
    if not isinstance(params_names, list):
        raise TypeError("Params names must be a list.")
    
    # Shape check.
    if x_0.shape != design_space.lower.shape:
        raise ValueError("Design space shape must be equal to x_0.")
    if len(params_names) != p:
        raise ValueError("Params size must be equal to p.")
    
    # Size check.
    if x_0.size == 0:
        raise ValueError("x_0 can't of size 0.")

    # Value Type check.
    if not isinstance(x_0[0], float):
        raise TypeError("x_0 must be an array of floats.")
    if not isinstance(params_names[0], str):
        raise TypeError("params names must be a list of strings.")

    # Value check.
    if not performance_criterion(f(x_0)):
        raise ValueError("x_0 must be a good design.")
    if N1 <= 0:
        raise ValueError("N1 must be positive.")
    if N2 <= 0:
        raise ValueError("N2 must be positive.")
    if growth_rate < 0:
        raise ValueError("Growth rate must be >= 0.")
    if confidence_lower < 0 or confidence_lower > 1:
        raise ValueError("Confidence interval must be between 0 and 1.")
    if confidence_upper < 0 or confidence_upper > 1:
        raise ValueError("Confidence interval must be between 0 and 1.")
    if confidence_lower >= confidence_upper:
        raise ValueError("Confidence bounds must be valid.")
    if alpha_c < 0 or alpha_c > 1:
        raise ValueError("alpha_c must be between 0 and 1.")
    if change_threshold < 0:
        raise ValueError("change_threshold must be > 0.")

    # Create initial candidate box around x_0.
    candidate_box = Box(x_0, x_0)

    # Set up output logs
    main_file = getattr(__main__, "__file__", None)
    output_dir = str(Path(main_file).parent) + "/outputs"
    fname = next_filename("run_{idx}_logs.csv", output_dir)
    header_cols = ["phase", "iteration", "size", "m", "N", "time"] + \
                  [f"lower_{i}" for i in range(p)] + \
                  [f"upper_{i}" for i in range(p)]
    with open(fname, "w") as file:
        print(",".join(header_cols), file=file)   
    
    # Phase I. While size of candidate box is changing:
    iterations_phase_1 = 0
    while True:
        iterations_phase_1 += 1
        size_prev = candidate_box.size()

        # Modification Step B: Extend candidate box.
        candidate_box = modification_step_B(candidate_box,
                                            design_space,
                                            growth_rate)

        # Compute Monte Carlo sample inside candidate box. Seperate into good
        # and bad designs, ordering bad designs in descending performance order.
        good_designs, bad_designs = monte_carlo_sample(N1, candidate_box, f,
                                                       performance_criterion)
        if good_designs is None or good_designs.size == 0:
            m = 0
        else:
            m = good_designs.shape[0]

        # Modification Step A: Remove bad sample designs.
        candidate_box = modification_step_A(candidate_box,
                                            good_designs,
                                            bad_designs)

        current_size = candidate_box.size()
        size_delta = current_size - size_prev

        # Record log
        time_stamp = perf_counter()
        elapsed_time = time_stamp - start_time
        row = [1, iterations_phase_1, candidate_box.size(), m, N1,
               elapsed_time] + \
              candidate_box.lower.tolist() + \
              candidate_box.upper.tolist()
        with open(fname, "a") as file:
            print(",".join(map(str, row)), file=file)

        # Terminate if size is not changing.
        if (abs(size_delta) < (change_threshold * size_prev)):
            break

    # Phase II. While ratio of good and bad designs does not satisfy confidence:
    iterations_phase_2 = 0
    while True:
        iterations_phase_2 += 1

        # Compute Monte Carlo sample in candidate box
        good_designs, bad_designs = monte_carlo_sample(N2, candidate_box, f,
                                                       performance_criterion)
        if good_designs is None or good_designs.size == 0:
            m = 0
        else:
            m = good_designs.shape[0]

        # Record log
        time_stamp = perf_counter()
        elapsed_time = time_stamp - start_time
        row = [2, iterations_phase_1 + iterations_phase_2, candidate_box.size(),
               m, N2, elapsed_time] + \
              candidate_box.lower.tolist() + \
              candidate_box.upper.tolist()
        with open(fname, "a") as file:
            print(",".join(map(str, row)), file=file)

        # Terminate if confidence is satisfied.
        current_confidence = (beta.cdf(confidence_upper, m + 1, N2 - m + 1) -
                              beta.cdf(confidence_lower, m + 1, N2 - m + 1))
        if (current_confidence >= 1.0 - alpha_c):
            break

        # Modification Step A: Remove bad sample designs.
        candidate_box = modification_step_A(candidate_box,
                                            good_designs,
                                            bad_designs)

    # Setup for plots
    df = pd.read_csv(fname)
    vol_design_space = design_space.size()
    df["normalized_volume"] = df["size"] / vol_design_space
    df["good_designs_fraction"] = df["m"] / df["N"]

    # Normalized volume plot
    fig, ax = plt.subplots()
    ax.plot(df["iteration"], df["normalized_volume"])
    ax.set_yscale("log")
    ax.set_xlabel("Iteration number")
    ax.set_ylabel("Normalized volume")
    ax.grid(True, which="both")
    x_div = iterations_phase_1 + 0.5
    ax.axvline(x=x_div, color="k", linewidth=2)
    xmax = df["iteration"].max()
    _, ymax = ax.get_ylim()
    ax.text((1 + iterations_phase_1) / 2.0, ymax, "Phase I",
            ha="center", va="bottom", fontsize=12)
    ax.text((x_div + xmax) / 2.0, ymax, "Phase II",
            ha="center", va="bottom", fontsize=12)
    out1 = next_filename("run_{idx}_normalized_vol.png", output_dir)
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Fraction of good designs plot
    fig, ax = plt.subplots()
    ax.plot(df["iteration"], df["good_designs_fraction"])
    ax.set_xlabel("Iteration number")
    ax.set_ylabel("Fraction of good designs")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, which="both")
    x_div = iterations_phase_1 + 0.5
    ax.axvline(x=x_div, color="k", linewidth=2)
    xmax = df["iteration"].max()
    _, ymax = ax.get_ylim()
    ax.text((1 + iterations_phase_1) / 2.0, ymax, "Phase I",
            ha="center", va="bottom", fontsize=12)
    ax.text((x_div + xmax) / 2.0, ymax, "Phase II",
            ha="center", va="bottom", fontsize=12)
    out2 = next_filename("run_{idx}_good_designs_fraction.png", output_dir)
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Bounds plot
    for i, name in enumerate(params_names):
        series_lower = df[f"lower_{i}"]
        series_upper = df[f"upper_{i}"]

        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(df["iteration"], series_lower, linewidth=2, label="Lower",
                color = 'blue')
        ax.plot(df["iteration"], series_upper, linewidth=2, label="Upper",
                color = 'blue')
        ax.fill_between(df["iteration"], series_lower, series_upper, alpha=0.15)
        lo, hi = float(design_space.lower[i]), float(design_space.upper[i])                     
        ax.set_ylim(lo, hi)              
        ax.set_title(f"Bounds over iterations - {name}\n")
        ax.set_xlabel("Iteration number")
        ax.set_ylabel("Bounds Value")
        ax.grid(True, which="both", alpha=0.3)
        ax.axvline(x=x_div, color="k", linewidth=2)
        _, ymax = ax.get_ylim()
        ax.text((1 + iterations_phase_1) / 2.0, ymax, "Phase I",
                ha="center", va="bottom", fontsize=10)
        ax.text((x_div + df["iteration"].max()) / 2.0, ymax, "Phase II",
                ha="center", va="bottom", fontsize=10)
        out_i = next_filename(f"run_{{idx}}_{name}.png", output_dir)
        fig.savefig(out_i, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return candidate_box