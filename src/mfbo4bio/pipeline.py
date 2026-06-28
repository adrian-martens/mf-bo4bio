from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import mfbo4bio.conditions_data as data
import mfbo4bio.virtual_lab as vl
from mfbo4bio.bo_core.engine import run_bo_engine
from mfbo4bio.config import RunConfig
from mfbo4bio.results import industrial_output_path, save_json
from mfbo4bio.utils import sampling


@dataclass(slots=True)
class IndustrialRunResult:
    payload: dict


def run_bo(config: RunConfig):
    if config.method not in {"qUCB", "qLogEI", "GIBBON"}:
        raise ValueError("run_bo supports qUCB, qLogEI, or GIBBON")
    return run_bo_engine(config)


def run_industrial(
    config: RunConfig,
    *,
    sampling_method: str = "latin_hypercube",
    platform_cond: bool = True,
) -> IndustrialRunResult:
    if config.method != "industrial":
        raise ValueError("run_industrial requires method='industrial'")

    if config.experiment.clone_distribution == "alpha":
        process_parameters = data.process_parameters_alpha
    elif config.experiment.clone_distribution == "beta":
        process_parameters = data.process_parameters_beta
    else:
        raise NameError("clone_distribution must be alpha or beta")

    rng = np.random.default_rng(config.seed)
    feeding_max = config.experiment.feeding_max
    mbr_level = config.experiment.mbr_level

    best_values = []
    best_from_batch = []
    fidelities = []
    batches = []
    cumulative_cost_list = [0]
    cumulative_cost = 0
    cell_clones_selected = []

    best_result_low = None
    best_result_mid = None

    t0 = 35
    ph0 = 7.2

    constraints = {
        "feeding_max": feeding_max,
        "feeding_dims": ["feeding1", "feeding2", "feeding3"],
        "mtp_feed_mode": config.experiment.mtp_feed_mode,
    }

    # Range of pH and Temp is chosen very narrowly on purpose
    # to mimic platform conditions in industrial application
    dimensions_dict = {
        "temperature": ("continuous", (t0 - 0.0001, t0)),
        "ph": ("continuous", (ph0 - 0.0001, ph0)),
        "feeding1": ("continuous", (0, feeding_max)),
        "feeding2": ("continuous", (0, feeding_max)),
        "feeding3": ("continuous", (0, feeding_max)),
        "fidelity": ("discrete", [0, mbr_level, 10]),
        "clone": ("discrete", list(range(len(process_parameters.keys())))),
    }

    num_batches_low = 5
    total_samples_low = num_batches_low * 12
    samples_low_all = sampling(
        method=sampling_method,
        dimensions_dict=dimensions_dict,
        num_samples=total_samples_low,
        constraints=constraints,
        fidelity_distribution={0: 1, mbr_level: 0, 10: 0},
        seed=rng,
    )
    low_batches = np.array_split(samples_low_all, num_batches_low)

    best_clone = 0
    for batch in low_batches:
        batches.append(batch)
        cell_clones_selected.append(batch[:, -1].tolist())
        results = vl.conduct_experiment(
            batch,
            mbr_level=mbr_level,
            clone_distribution=config.experiment.clone_distribution,
            rng=rng,
        )
        max_idx = int(np.argmax(results))
        best_sample = batch[max_idx]
        best_value = float(results[max_idx])
        best_from_batch.append(best_sample)
        fidelities.append(0)
        cumulative_cost += 10 * len(batch)
        cumulative_cost_list.append(cumulative_cost)
        best_values.append(best_value)
        if best_result_low is None or best_value > best_result_low:
            best_result_low = best_value
            best_clone = int(best_sample[-1])

    dimensions_dict.update(
        {
            "temperature": ("continuous", (30, 40)),
            "ph": ("continuous", (6, 8)),
            "clone": ("discrete", [best_clone]),
        }
    )

    num_batches_mid = 15
    total_samples_mid = num_batches_mid * 4
    samples_mid_all = sampling(
        method=sampling_method,
        dimensions_dict=dimensions_dict,
        num_samples=total_samples_mid,
        constraints=constraints,
        fidelity_distribution={0: 0, mbr_level: 1, 10: 0},
        seed=rng,
    )
    mid_batches = np.array_split(samples_mid_all, num_batches_mid)

    best_experiment = mid_batches[0][0]
    for batch in mid_batches:
        batches.append(batch)
        results = vl.conduct_experiment(
            batch,
            mbr_level=mbr_level,
            clone_distribution=config.experiment.clone_distribution,
            rng=rng,
        )
        max_idx = int(np.argmax(results))
        best_sample = batch[max_idx]
        best_value = float(results[max_idx])
        best_from_batch.append(best_sample)
        fidelities.append(mbr_level)
        cumulative_cost += 575 * len(batch)
        cumulative_cost_list.append(cumulative_cost)
        best_values.append(best_value)
        if best_result_mid is None or best_value > best_result_mid:
            best_result_mid = best_value
            best_experiment = best_sample

    t_val, ph_val, feed1, feed2, feed3, _, clone = best_experiment
    dimensions_dict.update(
        {
            "temperature": ("continuous", (t_val * 0.95, t_val * 1.05)),
            "ph": ("continuous", (ph_val * 0.95, ph_val * 1.05)),
            "feeding1": ("continuous", (feed1 * 0.95, feed1 * 1.05)),
            "feeding2": ("continuous", (feed2 * 0.95, feed2 * 1.05)),
            "feeding3": ("continuous", (feed3 * 0.95, feed3 * 1.05)),
            "clone": ("discrete", [int(clone)]),
        }
    )

    num_batches_high = 5
    samples_high_all = sampling(
        method=sampling_method,
        dimensions_dict=dimensions_dict,
        num_samples=num_batches_high,
        constraints=constraints,
        fidelity_distribution={0: 0, mbr_level: 0, 10: 1},
        seed=rng,
    )
    high_batches = np.array_split(samples_high_all, num_batches_high)

    for batch in high_batches:
        batches.append(batch)
        results = vl.conduct_experiment(
            batch,
            mbr_level=mbr_level,
            clone_distribution=config.experiment.clone_distribution,
            rng=rng,
        )
        max_idx = int(np.argmax(results))
        best_sample = batch[max_idx]
        best_value = float(results[max_idx])
        best_from_batch.append(best_sample)
        fidelities.append(10)
        cumulative_cost += 2100 * len(batch)
        cumulative_cost_list.append(cumulative_cost)
        best_values.append(best_value)

    flat_clones = [item for sublist in cell_clones_selected for item in sublist]
    payload = {
        "clone distribution": config.experiment.clone_distribution,
        "mbr_level": mbr_level,
        "n_iterations": len(best_values),
        "max_iterations": num_batches_low + num_batches_mid + num_batches_high,
        "best_values": best_values,
        "best_points": [point.tolist() for point in best_from_batch],
        "batches": [batch.tolist() for batch in batches],
        "cumulative_cost_list": cumulative_cost_list,
        "fidelities": fidelities,
        "clones": flat_clones,
        "X_mean": 0,
        "X_std": 1,
        "y_mean": 0,
        "y_std": 1,
        "error": None,
    }

    out = industrial_output_path(
        config, sampling_method=sampling_method, platform_cond=platform_cond
    )
    save_json(payload, out)
    return IndustrialRunResult(payload=payload)
