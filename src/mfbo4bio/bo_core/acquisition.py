from __future__ import annotations

from botorch.acquisition import qUpperConfidenceBound
from botorch.acquisition.cost_aware import GenericCostAwareUtility
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.acquisition.max_value_entropy_search import (
    qMultiFidelityLowerBoundMaxValueEntropy,
)
from botorch.acquisition.objective import IdentityMCObjective
from botorch.sampling import SobolQMCNormalSampler
from torch import Size

_IDENTITY_COST_UTILITY = GenericCostAwareUtility(cost=lambda X, deltas: deltas)


def build_acquisition(
    *,
    method: str,
    model,
    rng,
    best_f: float,
    beta: float,
    candidate_set,
    project_fn,
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
            cost_aware_utility=_IDENTITY_COST_UTILITY,
        )

    raise ValueError(f"Unknown BO method {method}")
