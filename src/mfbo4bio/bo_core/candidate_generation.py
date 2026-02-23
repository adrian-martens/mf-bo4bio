from __future__ import annotations

import numpy as np
import torch
from botorch.optim import optimize_acqf

from mfbo4bio.optimization import custom_optimization


def build_bounds(
    x_mean: np.ndarray, x_std: np.ndarray, num_clones: int, dtype=torch.float64
):
    lower = torch.tensor(
        (np.array([30.0, 6.0, 0.0, 0.0, 0.0, 0]) - x_mean) / x_std, dtype=dtype
    )
    lower = torch.hstack((lower, torch.tensor(0, dtype=dtype)))
    upper = torch.tensor(
        (np.array([40.0, 8.0, 50.0, 50.0, 50.0, 10]) - x_mean) / x_std, dtype=dtype
    )
    upper = torch.hstack((upper, torch.tensor(num_clones - 1, dtype=dtype)))
    return torch.stack([lower, upper])


def build_inequality_constraints(
    x_mean: np.ndarray,
    x_std: np.ndarray,
    feeding_max: float,
    tol: float = 0.4,
    dtype=torch.float64,
):
    mean_sum = x_mean[2:5].sum()
    std_sum = np.sqrt(x_std[2] ** 2 + x_std[3] ** 2 + x_std[4] ** 2)
    standardized_feeding_max = (feeding_max - mean_sum) / std_sum
    return [
        (
            torch.tensor([2, 3, 4], dtype=torch.long),
            torch.tensor([1.0, 1.0, 1.0], dtype=dtype),
            standardized_feeding_max - tol,
        ),
        (
            torch.tensor([2, 3, 4], dtype=torch.long),
            torch.tensor([-1.0, -1.0, -1.0], dtype=dtype),
            -standardized_feeding_max,
        ),
    ]


def propose_batch_for_fidelity(
    *,
    fidelity: int,
    mbr_level: int,
    batch_size: dict[int, int],
    acq_fn,
    bounds,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    rng,
    process_parameters,
):
    dtype = torch.float64
    fixed_features = {5: torch.tensor((fidelity - x_mean[5]) / x_std[5], dtype=dtype)}

    if fidelity == 0:
        candidate, acq_value = custom_optimization(
            x_mean,
            x_std,
            acq_fn,
            bounds,
            batch_size[fidelity],
            resolution=5,
            mode="sampling",
            seed=int(rng.integers(0, 2**32 - 1)),
            process_parameters=process_parameters,
        )
        return candidate, acq_value

    try:
        candidate, acq_value = optimize_acqf(
            acq_function=acq_fn,
            bounds=bounds,
            q=batch_size[fidelity],
            num_restarts=5,
            raw_samples=512,
            sequential=True,
            fixed_features=fixed_features,
            inequality_constraints=build_inequality_constraints(
                x_mean, x_std, feeding_max=50.0
            ),
        )
        return candidate, acq_value
    except RuntimeError as err:
        message = str(err)
        if "does not require grad" not in message:
            raise
        return _sampling_fallback_candidates(
            acq_fn=acq_fn,
            bounds=bounds,
            q=batch_size[fidelity],
            fixed_features=fixed_features,
            rng=rng,
        )


def _sampling_fallback_candidates(acq_fn, bounds, q, fixed_features, rng):
    dtype = torch.float64
    lower = bounds[0]
    upper = bounds[1]
    n_features = int(lower.numel())
    best_batch = None
    best_value = None

    for _ in range(64):
        u = torch.tensor(
            rng.random((q, n_features)),
            dtype=dtype,
        )
        batch = lower + (upper - lower) * u
        for dim, value in fixed_features.items():
            batch[:, dim] = value

        batch[:, -1] = torch.round(batch[:, -1])

        score = 0.0
        with torch.no_grad():
            for j in range(batch.shape[0]):
                x_j = batch[j].unsqueeze(0)
                x_pending = batch[:j] if j > 0 else None
                if hasattr(acq_fn, "set_X_pending"):
                    acq_fn.set_X_pending(x_pending)
                value_j = acq_fn(x_j)
                if torch.is_tensor(value_j):
                    score += float(value_j.sum().item())
                else:
                    score += float(value_j)
            if hasattr(acq_fn, "set_X_pending"):
                acq_fn.set_X_pending(None)

        if best_value is None or score > best_value:
            best_value = score
            best_batch = batch

    if best_batch is None:
        raise RuntimeError("Sampling fallback failed to generate candidates.")

    return best_batch, torch.tensor([best_value], dtype=dtype)
