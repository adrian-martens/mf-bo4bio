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
    dtype=torch.float64,
):
    # Min-max scaling: raw[i] = scaled[i] * x_std[i] + x_mean[i].
    # Constraint: raw[2] + raw[3] + raw[4] <= feeding_max
    # In scaled space: sum(scaled[i] * x_std[i]) <= feeding_max - sum(x_mean[2:5])
    # BoTorch format: coefficients @ X[indices] >= rhs  (>= convention)
    coeffs = torch.tensor(
        [-float(x_std[2]), -float(x_std[3]), -float(x_std[4])], dtype=dtype
    )
    rhs = -(feeding_max - float(x_mean[2:5].sum()))
    return [
        (torch.tensor([2, 3, 4], dtype=torch.long), coeffs, rhs),
    ]


def propose_batch_for_fidelity(
    *,
    fidelity: int,
    mbr_level: int,
    method: str = "qUCB",
    batch_size: dict[int, int],
    acq_fn,
    bounds,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    rng,
    process_parameters,
):
    dtype = torch.float64
    fidelity_fixed = torch.tensor((fidelity - x_mean[5]) / x_std[5], dtype=dtype)
    fixed_features = {5: fidelity_fixed}

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
        num_restarts = 10 if method == "GIBBON" else 5
        raw_samples = 1024 if method == "GIBBON" else 512
        candidate, acq_value = optimize_acqf(
            acq_function=acq_fn,
            bounds=bounds,
            q=batch_size[fidelity],
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            sequential=True,
            fixed_features=fixed_features,
            inequality_constraints=build_inequality_constraints(
                x_mean, x_std, feeding_max=50.0
            ),
            options={"maxiter": 300, "batch_limit": 5},
            retry_on_optimization_warning=True,
        )
        candidate = _clip_feeding_sum(candidate)
        if method == "GIBBON":
            # Keep clone/task ids valid integers after continuous optimization.
            max_clone = len(process_parameters.keys()) - 1
            candidate[:, -1] = torch.clamp(torch.round(candidate[:, -1]), 0, max_clone)
            acq_value = _evaluate_batch_acq(acq_fn=acq_fn, batch=candidate)
        acq_sum = (
            float(acq_value.sum().item())
            if torch.is_tensor(acq_value)
            else float(acq_value)
        )
        if not np.isfinite(acq_sum) or abs(acq_sum) < 1e-12:
            return _sampling_fallback_candidates(
                acq_fn=acq_fn,
                bounds=bounds,
                q=batch_size[fidelity],
                fixed_features=fixed_features,
                rng=rng,
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


def _evaluate_batch_acq(acq_fn, batch):
    values = []
    with torch.no_grad():
        for j in range(batch.shape[0]):
            x_j = batch[j].unsqueeze(0)
            x_pending = batch[:j] if j > 0 else None
            if hasattr(acq_fn, "set_X_pending"):
                acq_fn.set_X_pending(x_pending)
            value_j = acq_fn(x_j)
            values.append(
                float(value_j.sum().item())
                if torch.is_tensor(value_j)
                else float(value_j)
            )
        if hasattr(acq_fn, "set_X_pending"):
            acq_fn.set_X_pending(None)
    return torch.tensor(values, dtype=batch.dtype)


def _clip_feeding_sum(candidate: torch.Tensor) -> torch.Tensor:
    """Clip feeding dims (2,3,4) so their sum per row <= 1 in scaled space.

    With fixed bounds lower=[0,0,0] and upper=[50,50,50] for feeding,
    min-max scaling maps each dim to [0,1], so this enforces F1+F2+F3 <= 50
    in raw space. Underfeeding (sum < 50) is allowed.
    """
    feed = candidate[:, 2:5]
    feed_sum = feed.sum(dim=-1, keepdim=True)
    over = feed_sum > 1.0
    candidate = candidate.clone()
    candidate[:, 2:5] = torch.where(over, feed / feed_sum, feed)
    return candidate


def _sampling_fallback_candidates(acq_fn, bounds, q, fixed_features, rng):
    dtype = torch.float64
    lower = bounds[0]
    upper = bounds[1]
    n_features = int(lower.numel())
    best_batch = None
    best_value = None

    for _ in range(256):
        u = torch.tensor(
            rng.random((q, n_features)),
            dtype=dtype,
        )
        batch = lower + (upper - lower) * u
        for dim, value in fixed_features.items():
            batch[:, dim] = value

        batch = _clip_feeding_sum(batch)
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
