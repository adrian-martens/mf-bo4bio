from __future__ import annotations

import traceback
from dataclasses import asdict

import numpy as np
import torch

import mfbo4bio.conditions_data as data
import mfbo4bio.virtual_lab as vl
from mfbo4bio.bo_core.acquisition import build_acquisition
from mfbo4bio.bo_core.candidate_generation import (
    build_bounds,
    propose_batch_for_fidelity,
)
from mfbo4bio.bo_core.fidelity_selection import correlation_weighted_score, gibbon_score
from mfbo4bio.bo_core.state import BORunResult, BOState
from mfbo4bio.config import RunConfig
from mfbo4bio.randomness import SeedManager
from mfbo4bio.results import bo_output_paths, save_json
from mfbo4bio.utils import (
    destandardize_mixed_tensor,
    minmax_scale_data,
    sampling,
    standardize_mixed_tensor,
    standardize_target,
    train_gp_model,
)
from mfbo4bio.validation import (
    ensure_allowed_fidelities,
    ensure_finite,
    ensure_non_empty_2d,
    ensure_valid_clone_ids,
)


def _process_parameters(clone_distribution: str):
    if clone_distribution == "alpha":
        return data.process_parameters_alpha
    if clone_distribution == "beta":
        return data.process_parameters_beta
    raise NameError(
        f"No clone distribution named {clone_distribution}. \
        Please specify 'alpha' or 'beta'"
    )


def run_bo_engine(config: RunConfig) -> BORunResult:
    method = config.method
    exp_cfg = config.experiment
    bo_cfg = config.bo

    process_parameters = _process_parameters(exp_cfg.clone_distribution)
    rng = np.random.default_rng(config.seed)
    if config.reproducible and config.seed is not None:
        SeedManager(config.seed).torch_seed("global")

    dtype = torch.float64
    error_message = None

    best_values: list[float] = []
    best_from_batch: list[np.ndarray] = []
    batches: list[np.ndarray] = []
    fidelities: list[int] = [0]
    cumulative_cost_list: list[int] = [0]

    batch_size = bo_cfg.batch_size_by_fidelity or {0: 12, exp_cfg.mbr_level: 4, 10: 1}
    cost_level = bo_cfg.cost_by_fidelity or {0: 10, exp_cfg.mbr_level: 575, 10: 2100}

    xtrain = torch.empty(0, 7, dtype=dtype)
    ytrain = torch.empty(0, 1, dtype=dtype)
    x_mean = np.zeros(6)
    x_std = np.ones(6)
    y_mean = 0.0
    y_std = 1.0

    try:
        t0 = rng.uniform(30, 40)
        ph0 = rng.uniform(6, 8)
        dimensions_dict = {
            "temperature": ("continuous", (t0 - 0.0001, t0)),
            "ph": ("continuous", (ph0 - 0.0001, ph0)),
            "feeding1": ("continuous", (0, exp_cfg.feeding_max)),
            "feeding2": ("continuous", (0, exp_cfg.feeding_max)),
            "feeding3": ("continuous", (0, exp_cfg.feeding_max)),
            "fidelity": ("discrete", [0, exp_cfg.mbr_level, 10]),
            "clone": ("discrete", list(range(len(process_parameters.keys())))),
        }

        constraints = {
            "feeding_max": exp_cfg.feeding_max,
            "feeding_dims": ["feeding1", "feeding2", "feeding3"],
        }

        x_initial = sampling(
            method="latin_hypercube",
            dimensions_dict=dimensions_dict,
            num_samples=12,
            constraints=constraints,
            fidelity_distribution={0: 0.9, exp_cfg.mbr_level: 0.1, 10: 0},
            seed=rng,
        )

        ensure_non_empty_2d("X_initial", x_initial)
        ensure_allowed_fidelities(x_initial[:, -2], {0, exp_cfg.mbr_level, 10})
        ensure_valid_clone_ids(x_initial[:, -1], len(process_parameters.keys()))

        y_initial = np.asarray(
            vl.conduct_experiment(
                x_initial,
                clone_distribution=exp_cfg.clone_distribution,
                mbr_level=exp_cfg.mbr_level,
                rng=rng,
            ),
            dtype=float,
        )
        ensure_finite("y_initial", y_initial)

        best_initial = x_initial[np.argmax(y_initial)]
        x_cont = x_initial[:, :-1]
        x_cat = x_initial[:, -1].reshape(-1, 1)

        x_lower = np.array([30.0, 6.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        x_upper = np.array([40.0, 8.0, 50.0, 50.0, 50.0, 10.0], dtype=float)
        x_standardized, x_mean, x_std = minmax_scale_data(
            x_cont, X_lower=x_lower, X_upper=x_upper
        )
        y_standardized, y_mean, y_std = standardize_target(y_initial)

        xtrain = torch.tensor(np.hstack([x_standardized, x_cat]), dtype=dtype)
        ytrain = torch.tensor(y_standardized, dtype=dtype).unsqueeze(-1)

        state = BOState(
            x_all=x_initial,
            y_all=y_initial,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=float(y_mean),
            y_std=float(y_std),
            best_values=[float(ytrain.max().item())],
            best_points=[best_initial],
            batches=[x_initial],
            fidelities=[0],
            cumulative_cost_list=[0, cost_level[0] * 12],
        )

        if bo_cfg.task_representation == "HYBRID":
            model, _ = train_gp_model(
                xtrain,
                ytrain,
                model_type=bo_cfg.task_representation,
                embed_dim=bo_cfg.embed_dim,
            )
        else:
            model, _ = train_gp_model(
                xtrain, ytrain, model_type=bo_cfg.task_representation
            )

        fidelity_levels = [0, exp_cfg.mbr_level, 10]

        for iteration in range(bo_cfg.n_iterations):
            if state.cumulative_cost_list[-1] >= 40000:
                break

            bounds = build_bounds(
                x_mean, x_std, len(process_parameters.keys()), dtype=dtype
            )

            candidate_set_standardized = None
            project_to_fidelity = None
            if method == "GIBBON":
                candidate_set = sampling(
                    method="latin_hypercube",
                    dimensions_dict=dimensions_dict,
                    num_samples=512,
                    constraints=constraints,
                    fidelity_distribution={0: 0, exp_cfg.mbr_level: 0, 10: 1},
                    seed=rng,
                )
                cand_cont = candidate_set[:, :-1]
                cand_cat = torch.tensor(candidate_set[:, -1], dtype=dtype).unsqueeze(-1)
                candidate_set_standardized = (
                    torch.tensor(cand_cont, dtype=dtype) - x_mean
                ) / x_std
                candidate_set_standardized = torch.hstack(
                    (candidate_set_standardized, cand_cat)
                )

                def project_to_fidelity(x):
                    x = x.clone()
                    x[..., -2] = (10 - x_mean[-1]) / x_std[-1]
                    return x

            fidelity_std = float(x_std[5]) if abs(float(x_std[5])) > 1e-12 else 1.0
            standardized_cost_map = {
                float((fid - x_mean[5]) / fidelity_std): cost
                for fid, cost in cost_level.items()
            }

            acq_fn = build_acquisition(
                method=method,
                model=model,
                rng=rng,
                best_f=float(ytrain.max().item()),
                beta=bo_cfg.beta,
                candidate_set=candidate_set_standardized,
                project_fn=project_to_fidelity,
                cost_map=standardized_cost_map,
            )

            scored_candidates: dict[int, torch.Tensor] = {}
            scores: dict[int, torch.Tensor] = {
                0: torch.tensor(0.0),
                exp_cfg.mbr_level: torch.tensor(0.0),
                10: torch.tensor(0.0),
            }

            for fidelity in fidelity_levels:
                cand_std, acq_value = propose_batch_for_fidelity(
                    fidelity=fidelity,
                    mbr_level=exp_cfg.mbr_level,
                    method=method,
                    batch_size=batch_size,
                    acq_fn=acq_fn,
                    bounds=bounds,
                    x_mean=x_mean,
                    x_std=x_std,
                    rng=rng,
                    process_parameters=process_parameters,
                )
                scored_candidates[fidelity] = cand_std

                if method == "GIBBON":
                    scores[fidelity] = gibbon_score(
                        acq_value=acq_value,
                        cost=cost_level[fidelity],
                        batch_size=batch_size[fidelity],
                    )
                else:
                    scores[fidelity] = correlation_weighted_score(
                        model=model,
                        candidate_standardized=cand_std,
                        acq_value=acq_value,
                        cost=cost_level[fidelity],
                        batch_size=batch_size[fidelity],
                        x_mean=x_mean,
                        x_std=x_std,
                        logei=(method == "qLogEI"),
                    )

            selected_fidelity = max(scores, key=lambda fid: float(scores[fid].item()))
            if iteration == bo_cfg.n_iterations - 1:
                selected_fidelity = 10

            batch = (
                destandardize_mixed_tensor(
                    scored_candidates[selected_fidelity], x_mean, x_std
                )
                .detach()
                .numpy()
            )
            batch[:, -1] = np.round(batch[:, -1])

            new_y = np.asarray(
                vl.conduct_experiment(
                    batch,
                    clone_distribution=exp_cfg.clone_distribution,
                    mbr_level=exp_cfg.mbr_level,
                    rng=rng,
                ),
                dtype=float,
            )
            ensure_finite("new_y", new_y)

            state.append_batch(
                batch=batch,
                y_new=new_y,
                fidelity=selected_fidelity,
                batch_cost=batch_size[selected_fidelity]
                * cost_level[selected_fidelity],
            )

            x_cont = state.x_all[:, :-1]
            x_standardized, x_mean, x_std = minmax_scale_data(
                x_cont, X_lower=x_lower, X_upper=x_upper
            )
            y_standardized, y_mean, y_std = standardize_target(state.y_all)
            x_standardized_t = standardize_mixed_tensor(
                torch.tensor(state.x_all, dtype=dtype), x_mean, x_std
            )
            xtrain = x_standardized_t.clone().detach()
            ytrain = torch.tensor(y_standardized, dtype=dtype).unsqueeze(-1)

            if bo_cfg.task_representation == "HYBRID":
                model, _ = train_gp_model(
                    xtrain,
                    ytrain,
                    model_type=bo_cfg.task_representation,
                    embed_dim=bo_cfg.embed_dim,
                )
            else:
                model, _ = train_gp_model(
                    xtrain, ytrain, model_type=bo_cfg.task_representation
                )

        best_values = state.best_values
        best_from_batch = state.best_points
        batches = state.batches
        fidelities = state.fidelities
        cumulative_cost_list = state.cumulative_cost_list

    except Exception as exc:
        traceback.print_exc()
        error_message = str(exc)

    payload = {
        "clone distribution": exp_cfg.clone_distribution,
        "mbr_level": exp_cfg.mbr_level,
        "n_iterations": batch_size,
        "iterations": bo_cfg.n_iterations,
        "best_values": best_values,
        "batches": batches,
        "best_points": best_from_batch,
        "cumulative_cost_list": cumulative_cost_list,
        "fidelities": fidelities,
        "Xtrain": xtrain,
        "ytrain": ytrain,
        "X_mean": x_mean,
        "X_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "error": error_message,
        "config": asdict(config),
    }

    for out in bo_output_paths(config):
        save_json(payload, out)

    return BORunResult(payload=payload, x_train=xtrain, y_train=ytrain)
