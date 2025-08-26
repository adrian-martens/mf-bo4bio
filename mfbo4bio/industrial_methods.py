import json
import os
from itertools import product

import numpy as np
import torch
from scipy.stats import qmc
from scipy.stats.qmc import Sobol

import mfbo4bio.conditions_data as data
import mfbo4bio.virtual_lab as vl

feeding_max = data.feeding_max


def sampling(
    method="latin_hypercube",
    dimensions_dict={},
    num_samples=10,
    constraints=None,
    rng=None,
    fidelity_distribution=None,
):
    if rng is None:
        rng = np.random.default_rng()

    dimension_names = list(dimensions_dict.keys())
    other_dims = [d for d in dimension_names if d != "fidelity"]

    if method in {"latin_hypercube", "sobol"}:
        if method not in ["latin_hypercube", "sobol"]:
            raise ValueError(f"Unsupported sampling method: {method}")

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

    elif method == "factorial":
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

    else:
        raise ValueError(f"Unsupported sampling method: {method}")


def apply_feeding_constraints(samples, dimension_names, fidelity_value, constraints):
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
    """
    Simulates a DoE-based optimization process for cell culture.

    Parameters
    ----------
    n_iterations : int, optional
        The number of iterations to run the optimization. Defaults to 10.
    output_name : str, optional
        The base name for the output JSON file. Defaults to "latin_hypercube".
    sampling_method : str, optional
        The method to use for sampling experimental designs.
        Options include "latin_hypercube", "sobol", or "factorial".
        Defaults to "latin_hypercube".
    seed : int, optional
        The random seed for reproducibility. Defaults to None.
    clone_distribution : str, optional
        Specifies the cell clone distribution to use,
        either "alpha" or "beta". Defaults to "alpha".
    mbr_level : int, optional
        The medium-fidelity level to use in the simulation. Defaults to 7.
    platform_cond : bool, optional
        If True, locks the temperature and pH at the initial values (35 and 7)
        for the low-fidelity experiments.
        If False, these values are sampled. Defaults to True.

    Returns
    -------
    None
        The function does not return a value but saves the simulation results
        to a JSON file in the `./saved_runs/industrial` directory.

    """
    if clone_distribution == "alpha":
        process_parameters = data.process_parameters_alpha
    elif clone_distribution == "beta":
        process_parameters = data.process_parameters_beta
    else:
        raise NameError(
            f"No clone distribution named {clone_distribution}. \
                        Please specify 'alpha' or 'beta'."
        )

    try:
        best_values = []
        best_from_batch = []
        fidelities = []
        batches = []
        cumulative_cost_list = [0]
        cumulative_cost = 0
        cell_clones_selected = []

        best_result_low = None
        best_result_mid = None

        rng = np.random.default_rng(seed)

        T = 35  # 37
        pH = 7  # 7.2

        constraints = {
            "feeding_max": feeding_max,
            "feeding_dims": ["feeding1", "feeding2", "feeding3"],
        }

        # ===================== FIDELITY 0 =====================
        dimensions_dict = {
            "temperature": ("continuous", (T - 0.0001, T)),
            "ph": ("continuous", (pH - 0.0001, pH)),
            "feeding1": ("continuous", (0, feeding_max)),
            "feeding2": ("continuous", (0, feeding_max)),
            "feeding3": ("continuous", (0, feeding_max)),
            "fidelity": ("discrete", [0, mbr_level, 10]),
            "clone": ("discrete", list(range(len(process_parameters.keys())))),
        }

        num_batches_low = 5
        batch_size_low = 12
        total_samples_low = num_batches_low * batch_size_low

        fidelity_distribution = {0: 1, mbr_level: 0, 10: 0}
        samples_low_all = sampling(
            method=sampling_method,
            dimensions_dict=dimensions_dict,
            num_samples=total_samples_low,
            constraints=constraints,
            fidelity_distribution=fidelity_distribution,
            rng=rng,
        )
        low_batches = np.array_split(samples_low_all, num_batches_low)

        if sampling_method == "latin_hypercube":
            base_samples = qmc.LatinHypercube(d=2, seed=rng).random(num_batches_low)
            scaled_values = qmc.scale(base_samples, (30, 6), (40, 8))
            T_values = scaled_values[:, 0]
            pH_values = scaled_values[:, 1]

        elif sampling_method == "sobol":
            base_samples = Sobol(d=2, seed=rng).random(num_batches_low)
            scaled_values = qmc.scale(base_samples, (30, 6), (40, 8))
            T_values = scaled_values[:, 0]
            pH_values = scaled_values[:, 1]

        elif sampling_method == "factorial":
            levels_T = [30, 40]
            levels_pH = [6, 8]
            factorial_combinations = np.array(
                [[t, p] for t in levels_T for p in levels_pH]
            )

            # Shuffle combinations to avoid order bias
            rng.shuffle(factorial_combinations)
            T_values = factorial_combinations[:num_batches_low, 0]
            pH_values = factorial_combinations[:num_batches_low, 1]

        for i, batch in enumerate(low_batches):
            print(f"BATCH {i + 1} (FIDELITY 0)")

            if not platform_cond:
                batch[:, 0] = T_values[i]
                batch[:, 1] = pH_values[i]

            batches.append(batch)
            cell_clones_selected.append(batch[:, -1].tolist())
            results = vl.conduct_experiment(
                batch,
                mbr_level=mbr_level,
                clone_distribution=clone_distribution,
                rng=rng,
            )
            max_idx = np.argmax(results)
            best_sample = batch[max_idx]
            best_value = results[max_idx]
            best_from_batch.append(best_sample)
            fidelities.append(0)
            cumulative_cost += 10 * len(batch)
            cumulative_cost_list.append(cumulative_cost)
            best_values.append(best_value)

            if best_result_low is None or best_value > best_result_low:
                best_result_low = best_value
                best_clone = best_sample[-1]

        print(f"Best clone selected from fidelity 0: {best_clone}")

        # ===================== FIDELITY mbr_level =====================
        dimensions_dict.update(
            {
                "temperature": ("continuous", (30, 40)),
                "ph": ("continuous", (6, 8)),
                "clone": ("discrete", [int(best_clone)]),
            }
        )

        num_batches_mid = 15  # 10
        batch_size_mid = 4
        total_samples_mid = num_batches_mid * batch_size_mid

        fidelity_distribution = {0: 0, mbr_level: 1, 10: 0}
        samples_mid_all = sampling(
            method=sampling_method,
            dimensions_dict=dimensions_dict,
            num_samples=total_samples_mid,
            constraints=constraints,
            fidelity_distribution=fidelity_distribution,
            rng=rng,
        )
        mid_batches = np.array_split(samples_mid_all, num_batches_mid)

        for i, batch in enumerate(mid_batches):
            print(f"BATCH {i + 1} (FIDELITY {mbr_level})")
            batches.append(batch)
            results = vl.conduct_experiment(
                batch,
                mbr_level=mbr_level,
                clone_distribution=clone_distribution,
                rng=rng,
            )
            max_idx = np.argmax(results)
            best_sample = batch[max_idx]
            best_value = results[max_idx]
            best_from_batch.append(best_sample)
            fidelities.append(mbr_level)
            cumulative_cost += 575 * len(batch)
            cumulative_cost_list.append(cumulative_cost)
            best_values.append(best_value)

            if best_result_mid is None or best_value > best_result_mid:
                best_result_mid = best_value
                best_experiment = best_sample

        # ===================== FIDELITY 10 =====================
        T, pH, feed1, feed2, feed3, _, clone = best_experiment

        dimensions_dict.update(
            {
                "temperature": ("continuous", (T * 0.95, T * 1.05)),
                "ph": ("continuous", (pH * 0.95, pH * 1.05)),
                "feeding1": ("continuous", (feed1 * 0.95, feed1 * 1.05)),
                "feeding2": ("continuous", (feed2 * 0.95, feed2 * 1.05)),
                "feeding3": ("continuous", (feed3 * 0.95, feed3 * 1.05)),
                "clone": ("discrete", [int(clone)]),
            }
        )

        num_batches_high = 5  # 3
        batch_size_high = 1
        total_samples_high = num_batches_high * batch_size_high

        fidelity_distribution = {0: 0, mbr_level: 0, 10: 1}
        samples_high_all = sampling(
            method=sampling_method,
            dimensions_dict=dimensions_dict,
            num_samples=total_samples_high,
            constraints=constraints,
            fidelity_distribution=fidelity_distribution,
            rng=rng,
        )
        high_batches = np.array_split(samples_high_all, num_batches_high)

        for i, batch in enumerate(high_batches):
            print(f"BATCH {i + 1} (FIDELITY 10)")
            batches.append(batch)
            results = vl.conduct_experiment(
                batch,
                mbr_level=mbr_level,
                clone_distribution=clone_distribution,
                rng=rng,
            )
            max_idx = np.argmax(results)
            best_sample = batch[max_idx]
            best_value = results[max_idx]
            best_from_batch.append(best_sample)
            fidelities.append(10)
            cumulative_cost += 2100 * len(batch)
            cumulative_cost_list.append(cumulative_cost)
            best_values.append(best_value)

    finally:
        cell_clones_selected = [
            item for sublist in cell_clones_selected for item in sublist
        ]
        clean_points = [
            (
                point.tolist()
                if isinstance(point, (np.ndarray, torch.Tensor))
                else list(point)
            )
            for point in best_from_batch
        ]

        bo_results = {
            "clone distribution": clone_distribution,
            "mbr_level": mbr_level,
            "n_iterations": 15,
            "iterations": n_iterations,
            "best_values": best_values,
            "best_points": clean_points,
            "batches": [batch.tolist() for batch in batches],
            "cumulative_cost_list": cumulative_cost_list,
            "fidelities": fidelities,
            "clones": cell_clones_selected,
            "X_mean": 0,
            "X_std": 1,
            "y_mean": 0,
            "y_std": 1,
            "error": None,
        }

        cond = "platform" if platform_cond else "var"

        os.makedirs(
            f"./saved_runs/industrial/{clone_distribution}_{cond} \
                    /{sampling_method}",
            exist_ok=True,
        )
        with open(
            f"./saved_runs/industrial/{clone_distribution}_{cond}/ \
                  {sampling_method}/{output_name}.json",
            "w",
        ) as f:
            json.dump(bo_results, f, indent=4)
