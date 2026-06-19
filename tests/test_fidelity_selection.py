import pytest
import torch

from mfbo4bio.bo_core.fidelity_selection import correlation_weighted_score, gibbon_score


class _Posterior:
    def __init__(self, cov):
        self.covariance_matrix = cov


class _Model:
    def eval(self):
        return self

    def __call__(self, x):
        n = x.shape[0] // 2
        cov = torch.eye(2 * n, dtype=torch.float64)
        cov[:n, n:] = 0.5 * torch.eye(n, dtype=torch.float64)
        cov[n:, :n] = 0.5 * torch.eye(n, dtype=torch.float64)
        return _Posterior(cov)


def test_correlation_weighted_score_positive():
    model = _Model()
    cand = torch.ones((2, 7), dtype=torch.float64)
    acq = torch.tensor([1.0, 2.0], dtype=torch.float64)
    score = correlation_weighted_score(
        model=model,
        candidate_standardized=cand,
        acq_value=acq,
        cost=10,
        batch_size=2,
        x_mean=torch.zeros(6),
        x_std=torch.ones(6),
        logei=False,
    )
    assert score.item() > 0


def test_gibbon_score_positive():
    score = gibbon_score(acq_value=torch.tensor([1.0]), cost=10, batch_size=1)
    assert score.item() == pytest.approx(0.1)
