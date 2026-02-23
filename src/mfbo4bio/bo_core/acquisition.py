from __future__ import annotations

from collections.abc import Mapping

import torch
from botorch.acquisition import qUpperConfidenceBound
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.max_value_entropy_search import (
    qMultiFidelityLowerBoundMaxValueEntropy,
)
from botorch.acquisition.objective import IdentityMCObjective
from botorch.models.deterministic import DeterministicModel
from botorch.sampling import SobolQMCNormalSampler
from torch import Size


class FixedFidelityCostModel(DeterministicModel):
    def __init__(
        self,
        cost_map: Mapping[float, int],
        fidelity_dim: int = -2,
    ) -> None:
        super().__init__()
        self.fidelity_dim = fidelity_dim
        self._num_outputs = 1
        fidelity_values = sorted(cost_map)
        self.register_buffer("fidelities", torch.tensor(fidelity_values))
        self.register_buffer(
            "costs",
            torch.tensor([cost_map[f] for f in fidelity_values], dtype=torch.float),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        fidelity_vals = X[..., self.fidelity_dim]
        # Robustly map possibly standardized fidelities to the nearest known level.
        distances = torch.abs(fidelity_vals.unsqueeze(-1) - self.fidelities)
        nearest_idx = torch.argmin(distances, dim=-1)
        return self.costs[nearest_idx].unsqueeze(-1)

    @property
    def num_outputs(self) -> int:
        return self._num_outputs


def build_acquisition(
    *,
    method: str,
    model,
    rng,
    best_f: float,
    beta: float,
    candidate_set,
    project_fn,
    cost_map: dict[float, int],
):
    if method == "qUCB":
        return qUpperConfidenceBound(
            model=model,
            beta=beta,
            sampler=SobolQMCNormalSampler(
                sample_shape=Size([128]), seed=int(rng.integers(0, 2**32 - 1))
            ),
            objective=IdentityMCObjective(),
        )

    if method == "qLogEI":
        return qLogExpectedImprovement(
            model=model,
            best_f=best_f,
            sampler=SobolQMCNormalSampler(
                sample_shape=Size([128]), seed=int(rng.integers(0, 2**32 - 1))
            ),
            objective=IdentityMCObjective(),
        )

    if method == "GIBBON":
        return qMultiFidelityLowerBoundMaxValueEntropy(
            model=model,
            candidate_set=candidate_set,
            project=project_fn,
            use_gumbel=True,
            cost_aware_utility=InverseCostWeightedUtility(
                cost_model=FixedFidelityCostModel(cost_map)
            ),
        )

    raise ValueError(f"Unknown BO method {method}")
