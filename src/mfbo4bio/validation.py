from __future__ import annotations

import numpy as np
import torch


def ensure_non_empty_2d(name: str, value: np.ndarray) -> None:
    if value.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {value.shape}")
    if value.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")


def ensure_finite(name: str, value: np.ndarray | torch.Tensor) -> None:
    if isinstance(value, np.ndarray):
        ok = np.isfinite(value).all()
    else:
        ok = torch.isfinite(value).all().item()
    if not ok:
        raise ValueError(f"{name} contains non-finite values")


def ensure_valid_clone_ids(clone_ids: np.ndarray, n_clones: int) -> None:
    rounded = np.round(clone_ids).astype(int)
    if (rounded < 0).any() or (rounded >= n_clones).any():
        raise ValueError("clone IDs out of valid range")


def ensure_allowed_fidelities(fidelity: np.ndarray, allowed: set[int]) -> None:
    rounded = np.round(fidelity).astype(int)
    bad = sorted(set(rounded.tolist()) - allowed)
    if bad:
        raise ValueError(f"invalid fidelity values: {bad}; allowed={sorted(allowed)}")
