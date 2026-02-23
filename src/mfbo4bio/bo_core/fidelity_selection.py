from __future__ import annotations

import torch


def correlation_weighted_score(
    *,
    model,
    candidate_standardized: torch.Tensor,
    acq_value,
    cost: int,
    batch_size: int,
    x_mean,
    x_std,
    logei: bool,
) -> torch.Tensor:
    candidate_high = candidate_standardized.clone().detach()
    candidate_high[:, -2] = torch.ones_like(candidate_high[:, -2]) * (
        (10 - x_mean[-1]) / x_std[-1]
    )

    comparison = torch.vstack((candidate_standardized, candidate_high))
    model.eval()
    with torch.no_grad():
        posterior = model(comparison)
        cov_matrix = posterior.covariance_matrix

    n = candidate_standardized.shape[0]
    var_low = cov_matrix[:n, :n].diagonal()
    var_high = cov_matrix[n:, n:].diagonal()
    covs = cov_matrix[:n, n:].diagonal()
    correlations = covs / torch.sqrt(var_low * var_high)

    base = torch.exp(acq_value) if logei else acq_value
    return ((base * correlations) / (batch_size * cost)).sum()


def gibbon_score(*, acq_value, cost: int, batch_size: int) -> torch.Tensor:
    return (acq_value).sum()  # (acq_value / (batch_size * cost)).sum()
