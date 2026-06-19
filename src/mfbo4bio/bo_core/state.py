from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass(slots=True)
class BOState:
    x_all: np.ndarray
    y_all: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float
    best_values: list[float] = field(default_factory=list)
    best_points: list[np.ndarray] = field(default_factory=list)
    batches: list[np.ndarray] = field(default_factory=list)
    fidelities: list[int] = field(default_factory=list)
    cumulative_cost_list: list[int] = field(default_factory=lambda: [0])

    def append_batch(
        self, batch: np.ndarray, y_new: np.ndarray, fidelity: int, batch_cost: int
    ) -> None:
        self.x_all = np.vstack([self.x_all, batch])
        self.y_all = np.concatenate([self.y_all, y_new])
        self.batches.append(batch)
        self.fidelities.append(int(fidelity))
        self.best_values.append(float(np.max(self.y_all)))
        self.best_points.append(batch[np.argmax(y_new)])
        self.cumulative_cost_list.append(
            self.cumulative_cost_list[-1] + int(batch_cost)
        )


@dataclass(slots=True)
class BORunResult:
    payload: dict
    x_train: torch.Tensor
    y_train: torch.Tensor
