from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mfbo4bio.config import RunConfig


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    return value


def bo_output_paths(config: RunConfig) -> list[Path]:
    root = project_root() / config.results_root
    clone = config.experiment.clone_distribution
    method = config.method
    if method == "GIBBON":
        # Keep output directory aligned with chosen task representation.
        # HYBRID runs are stored under "GIBBON", while legacy ICM runs are
        # stored under "gibbon_icm" for backward compatibility.
        if config.bo.task_representation == "ICM_WRAPPED":
            return [root / "bo" / clone / "gibbon_icm" / f"{config.output_name}.json"]
        return [root / "bo" / clone / "GIBBON" / f"{config.output_name}.json"]
    return [root / "bo" / clone / method / f"{config.output_name}.json"]


def industrial_output_path(
    config: RunConfig, sampling_method: str, platform_cond: bool
) -> Path:
    root = project_root() / config.results_root
    cond = "platform" if platform_cond else "var"
    clone = config.experiment.clone_distribution
    return (
        root
        / "industrial"
        / f"{clone}_{cond}"
        / sampling_method
        / f"{config.output_name}.json"
    )


def save_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(_to_jsonable(payload), file, indent=4)
