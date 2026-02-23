from itertools import product

import numpy as np
from scipy.stats import qmc
from scipy.stats.qmc import Sobol

import mfbo4bio.conditions_data as data

feeding_max = data.feeding_max


def factorial(
    dimensions_dict={},
    num_samples=10,
    constraints=None,
    rng=None,
    fidelity_distribution=None,
):
    """
    Generates a full factorial experimental design.

    This function creates a full factorial design by generating all possible
    combinations of discrete levels for each dimension. It is most suitable for
    discrete variables but can also handle continuous variables by binarizing
    them into a few distinct levels.

    Parameters
    ----------
    dimensions_dict : dict
        A dictionary defining the dimensions of the search space. Keys are dimension
        names, and values are tuples of (type, values). Continuous dimensions are
        binarized into three levels (min, mid, max).
    num_samples : int, optional
        The total number of samples to generate. Note that the actual number of
        samples generated might be lower if `num_samples` is less than the total
        number of factorial combinations. Defaults to 10.
    constraints : dict, optional
        A dictionary specifying any constraints on the dimensions. This is
        primarily used for "feeding" variables. Defaults to None.
    rng : numpy.random.Generator, optional
        A NumPy random number generator for reproducibility. If None, a new one
        is created. Defaults to None.
    fidelity_distribution : dict, optional
        A dictionary mapping fidelity levels to their desired proportion in the
        sample set (e.g., `{0: 0.8, 1: 0.2}`). This is required to match
        the generated samples to the desired fidelity distribution.

    Returns
    -------
    numpy.ndarray
        A 2D NumPy array of shape `(num_samples, num_dimensions)` containing the
        generated samples.

    Raises
    ------
    ValueError
        If the `fidelity_distribution` is not provided or if a dimension type
        is not "continuous" or "discrete".

    Notes
    -----
    - This method can quickly lead to a very large number of samples as the number
        of dimensions or levels increases, making it impractical for
        high-dimensional problems.
    """

    dimension_names = list(dimensions_dict.keys())
    other_dims = [d for d in dimension_names if d != "fidelity"]

    # Full factorial: Only works for discrete or binarized continuous dimensions
    levels = []
    for dim in other_dims:
        dtype, values = dimensions_dict[dim]
        if dtype == "continuous":
            levels.append(
                [values[0], (values[0] + values[1]) / 2, values[1]]
            )  # 2-level version
        elif dtype == "discrete":
            levels.append(values)
        else:
            raise ValueError(f"Unsupported type in factorial: {dtype}")

    factorial_samples = np.array(list(product(*levels)))
    rng.shuffle(factorial_samples)

    if fidelity_distribution is None:
        raise ValueError("fidelity_distribution must be provided.")

    # Match fidelity distribution
    fidelity_values = list(fidelity_distribution.keys())
    proportions = list(fidelity_distribution.values())
    counts = {
        f: int(round(num_samples * p)) for f, p in zip(fidelity_values, proportions)
    }

    # Fix count rounding
    total = sum(counts.values())
    while total != num_samples:
        k = max(counts, key=counts.get)
        counts[k] += num_samples - total
        total = sum(counts.values())

    final_samples = []
    for fidelity, count in counts.items():
        if count == 0:
            continue
        selected = factorial_samples[:count]
        full = np.zeros((count, len(dimension_names)))
        for i, d in enumerate(other_dims):
            full[:, dimension_names.index(d)] = selected[:, i]
        full[:, dimension_names.index("fidelity")] = fidelity

        if constraints:
            full = apply_feeding_constraints(
                full, dimension_names, fidelity, constraints
            )

        final_samples.append(full)

    return np.vstack(final_samples)[:num_samples]


def quasi_mc_methods(
    method,
    dimensions_dict={},
    num_samples=10,
    constraints=None,
    rng=None,
    fidelity_distribution=None,
):
    """
    Generates quasi-Monte Carlo (QMC) experimental design samples.

    This function supports Latin Hypercube and Sobol sampling methods. It
    distributes samples across different fidelity levels according to a
    specified probability distribution and handles both continuous and
    discrete dimensions.

    Parameters
    ----------
    method : str
        The QMC sampling method to use. Must be either "latin_hypercube" or "sobol".
    dimensions_dict : dict
        A dictionary defining the dimensions of the search space. Keys are dimension
        names, and values are tuples of (type, values), e.g.,
        `{"pH": ("continuous", [7.0, 7.5])}`.
    num_samples : int, optional
        The total number of samples to generate. Defaults to 10.
    constraints : dict, optional
        A dictionary specifying constraints on the dimensions, such as total
        feeding limits. Defaults to None.
    rng : numpy.random.Generator, optional
        A NumPy random number generator for reproducibility. If None, a new
        one is created. Defaults to None.
    fidelity_distribution : dict, optional
        A dictionary mapping fidelity levels to their desired proportions in the
        final sample set (e.g., `{0: 0.8, 1: 0.2}`).

    Returns
    -------
    numpy.ndarray
        A 2D NumPy array of shape `(num_samples, num_dimensions)` containing
        the generated samples.

    Raises
    ------
    ValueError
        If the `method` is unsupported, the "fidelity" dimension is not
        defined as discrete, or `fidelity_distribution` is not provided.

    Notes
    -----
    - The "fidelity" dimension must be included in `dimensions_dict` and must
        be a discrete variable for this function to work correctly.
    """

    if method in {"latin_hypercube", "sobol"}:

        if (
            "fidelity" not in dimensions_dict
            or dimensions_dict["fidelity"][0] != "discrete"
        ):
            raise ValueError(
                "fidelity must be a discrete variable for distribution control."
            )

        if fidelity_distribution is None:
            raise ValueError("fidelity_distribution must be provided.")

        dimension_names = list(dimensions_dict.keys())
        other_dims = [d for d in dimension_names if d != "fidelity"]
        num_dimensions = len(other_dims)

        # Determine how many samples per fidelity value
        fidelity_values = list(fidelity_distribution.keys())
        fidelity_proportions = list(fidelity_distribution.values())
        fidelity_sample_counts = {
            d: int(round(num_samples * p))
            for d, p in zip(fidelity_values, fidelity_proportions)
        }

        total_assigned = sum(fidelity_sample_counts.values())
        while total_assigned != num_samples:
            max_key = max(fidelity_sample_counts, key=fidelity_sample_counts.get)
            fidelity_sample_counts[max_key] += num_samples - total_assigned
            total_assigned = sum(fidelity_sample_counts.values())

        final_samples = []

        for fidelity_value, count in fidelity_sample_counts.items():
            if count == 0:
                continue

            # Initialize sampler
            if method == "latin_hypercube":
                sampler = qmc.LatinHypercube(d=num_dimensions, seed=rng)
            elif method == "sobol":
                sampler = Sobol(d=num_dimensions, scramble=True, seed=rng)
            base_samples = sampler.random(n=count)

            scaled_samples = np.zeros((count, len(dimension_names)))

            for i, dim_name in enumerate(other_dims):
                dim_type, dim_values = dimensions_dict[dim_name]
                dim_idx = dimension_names.index(dim_name)

                if dim_type == "continuous":
                    low, high = dim_values
                    scaled_samples[:, dim_idx] = qmc.scale(
                        base_samples[:, [i]], low, high
                    ).flatten()
                elif dim_type == "discrete":
                    allowed_values = np.array(dim_values)
                    indices = (
                        np.floor(
                            qmc.scale(base_samples[:, [i]], 0, len(allowed_values))
                        )
                        .astype(int)
                        .flatten()
                    )
                    indices = np.clip(indices, 0, len(allowed_values) - 1)
                    scaled_samples[:, dim_idx] = allowed_values[indices]
                else:
                    raise ValueError(f"Invalid dimension type: {dim_type}")

            # Assign fidelity value
            fidelity_index = dimension_names.index("fidelity")
            scaled_samples[:, fidelity_index] = fidelity_value

            # Apply constraints
            if constraints:
                feeding_dims = constraints.get("feeding_dims", [])
                feeding_max = constraints.get("feeding_max", 0)

                if fidelity_value == 0:
                    for row in scaled_samples:
                        for dim in feeding_dims:
                            if dim in dimension_names:
                                row[dimension_names.index(dim)] = 0
                        if "feeding2" in dimension_names:
                            row[dimension_names.index("feeding2")] = feeding_max
                else:
                    feeding_indices = [
                        dimension_names.index(dim)
                        for dim in feeding_dims
                        if dim in dimension_names
                    ]
                    if feeding_indices:
                        feeding_values = np.sum(
                            scaled_samples[:, feeding_indices], axis=1, keepdims=True
                        )
                        feeding_values[feeding_values == 0] = (
                            1  # Avoid division by zero
                        )
                        scaled_samples[:, feeding_indices] *= (
                            feeding_max / feeding_values
                        )

            final_samples.append(scaled_samples)

        final_samples = np.vstack(final_samples)
        return final_samples[:num_samples]


def sampling(
    method="latin_hypercube",
    dimensions_dict={},
    num_samples=10,
    constraints=None,
    rng=None,
    fidelity_distribution=None,
):
    """
    Generates a set of experimental design samples for Bayesian Optimization.

    This function supports various sampling methods, including Latin Hypercube, Sobol,
    and Full Factorial, to generate a set of experiments based on the specified
    dimensions and constraints. It can handle both continuous and discrete
    variables and applies a user-defined fidelity distribution.

    Parameters
    ----------
    method : str, optional
        The sampling method to use. Options are "latin_hypercube", "sobol", or
        "factorial". Defaults to "latin_hypercube".
    dimensions_dict : dict
        A dictionary defining the dimensions of the search space. Each key is the
        dimension name (e.g., "pH", "temp", "feeding1"), and the value is a tuple
        containing the dimension type ("continuous" or "discrete") and its allowed
        values (e.g., `(7, 7.5)` or `[0, 10, 20]`).
    num_samples : int, optional
        The total number of samples to generate. Defaults to 10.
    constraints : dict, optional
        A dictionary specifying any constraints on the dimensions. This is
        primarily used for "feeding" variables, where the total sum of
        feeding amounts is constrained. Defaults to None.
    rng : numpy.random.Generator, optional
        A NumPy random number generator for reproducibility. If None, a new one
        is created. Defaults to None.
    fidelity_distribution : dict, optional
        A dictionary mapping fidelity levels to their desired proportion in the
        sample set (e.g., `{0: 0.8, 1: 0.2}`). Required for
        "latin_hypercube" and "sobol" methods. Defaults to None.

    Returns
    -------
    numpy.ndarray
        A 2D NumPy array of shape (num_samples, num_dimensions) containing the
        generated samples. Each row represents a single experiment.

    Raises
    ------
    ValueError
        If the sampling `method` is unsupported, the `fidelity` dimension is
        not specified as discrete, or `fidelity_distribution` is not provided
        when required.

    Example
    -------
        rng = np.random.default_rng(seed)

        T = 35
        pH = 7

        constraints = {
            "feeding_max": feeding_max,
            "feeding_dims": ["feeding1", "feeding2", "feeding3"],
        }

        dimensions_dict = {
            "temperature": ("continuous", (T - 10, T + 10)),
            "ph": ("continuous", (pH - 3, pH + 2)),
            "feeding1": ("continuous", (0, feeding_max)),
            "feeding2": ("continuous", (0, feeding_max)),
            "feeding3": ("continuous", (0, feeding_max)),
            "fidelity": ("discrete", [0, 7, 10]),
            "clone": ("discrete", list(range(30))),
        }

        fidelity_distribution = {0: 0.25, 7: 0.5, 10: 0.25}

        samples_low_all = sampling(
            method=sampling_method,
            dimensions_dict=dimensions_dict,
            num_samples=10,
            constraints=constraints,
            fidelity_distribution=fidelity_distribution,
            rng=rng,
        )

    """

    if rng is None:
        rng = np.random.default_rng()

    if method in {"latin_hypercube", "sobol"}:

        return quasi_mc_methods(
            method,
            dimensions_dict,
            num_samples,
            constraints,
            rng,
            fidelity_distribution,
        )

    elif method == "factorial":

        return factorial(
            dimensions_dict, num_samples, constraints, rng, fidelity_distribution
        )

    else:
        raise ValueError(f"Unsupported sampling method: {method}")


def apply_feeding_constraints(samples, dimension_names, fidelity_value, constraints):
    """
    Applies constraints to 'feeding' dimensions based on fidelity level.

    This function modifies the 'feeding' variables in a sample set to ensure they
    meet specific constraints, which vary depending on the experiment's fidelity level.
    For low-fidelity experiments (fidelity == 0), feeding is fixed, while for
    high-fidelity experiments, the sum of feeding values is normalized to a maximum.

    Parameters
    ----------
    samples : numpy.ndarray
        A 2D NumPy array of shape `(n_samples, n_dimensions)` containing the
        experimental design points.
    dimension_names : list of str
        A list of strings containing the names of the dimensions in the same order as
        they appear in `samples`.
    fidelity_value : int or float
        The fidelity level of the current samples being processed. Typically 0 for
        low-fidelity and a non-zero value for high-fidelity.
    constraints : dict
        A dictionary containing the constraint parameters, specifically:
        - `feeding_dims` : list of str
            The names of the dimensions representing the feeding variables.
        - `feeding_max` : float
            The maximum allowed sum for the feeding variables in high-fidelity
            experiments.

    Returns
    -------
    numpy.ndarray
        The modified `samples` array with feeding constraints applied.

    Notes
    -----
    - For `fidelity_value == 0`, the first feeding dimension is set to 0 and the
        second is set to `feeding_max`. This assumes a fixed feeding schedule for
        low-fidelity runs.
    - For `fidelity_value != 0`, the feeding values are scaled proportionally so
        that their sum equals `feeding_max`.
    """

    feeding_dims = constraints.get("feeding_dims", [])
    feeding_max = constraints.get("feeding_max", 0)
    feeding_indices = [
        dimension_names.index(d) for d in feeding_dims if d in dimension_names
    ]

    if fidelity_value == 0:
        for row in samples:
            for d in feeding_dims:
                idx = dimension_names.index(d)
                if d == "feeding2":
                    row[idx] = feeding_max
                else:
                    row[idx] = 0
    else:
        feed_vals = np.sum(samples[:, feeding_indices], axis=1, keepdims=True)
        feed_vals[feed_vals == 0] = 1  # prevent div by zero
        samples[:, feeding_indices] *= feeding_max / feed_vals

    return samples


def run(
    n_iterations=10,
    output_name="latin_hypercube",
    sampling_method="latin_hypercube",
    seed=None,
    clone_distribution="alpha",
    mbr_level=7,
    platform_cond=True,
):
    from mfbo4bio.config import BOConfig, ExperimentConfig, RunConfig
    from mfbo4bio.pipeline import run_industrial

    cfg = RunConfig(
        method="industrial",
        output_name=output_name,
        seed=seed,
        experiment=ExperimentConfig(
            clone_distribution=clone_distribution,
            mbr_level=int(mbr_level),
        ),
        bo=BOConfig(n_iterations=n_iterations),
    )
    run_industrial(cfg, sampling_method=sampling_method, platform_cond=platform_cond)
