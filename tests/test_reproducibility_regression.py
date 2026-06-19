import json
from pathlib import Path

import numpy as np
import torch

from mfbo4bio.config import BOConfig, ExperimentConfig, RunConfig
from mfbo4bio.pipeline import run_bo


class _DummyPosterior:
    def __init__(self, n):
        self.covariance_matrix = torch.eye(n, dtype=torch.float64)


class _DummyModel:
    def eval(self):
        return self

    def __call__(self, x):
        return _DummyPosterior(x.shape[0])


class _DummyAcq:
    def __call__(self, x):
        return torch.ones((x.shape[0],), dtype=torch.float64)

    def set_X_pending(self, x):
        return None


def _fake_sampling(*args, **kwargs):
    num_samples = kwargs["num_samples"]
    arr = np.zeros((num_samples, 7), dtype=float)
    arr[:, 0] = 35.0
    arr[:, 1] = 7.0
    arr[:, 2] = 0.0
    arr[:, 3] = 50.0
    arr[:, 4] = 0.0
    arr[:, 5] = 0.0
    arr[:, 6] = np.arange(num_samples) % 3
    return arr


def _fake_experiment(x, **kwargs):
    x = np.asarray(x)
    return (x[:, 3] + x[:, 6] + 1).tolist()


def _fake_train_gp_model(*args, **kwargs):
    return _DummyModel(), None


def _fake_build_acquisition(**kwargs):
    return _DummyAcq()


def _fake_propose_batch_for_fidelity(**kwargs):
    fidelity = kwargs["fidelity"]
    candidate = torch.tensor(
        [[35.0, 7.0, 0.0, 50.0, 0.0, float(fidelity), 1.0]], dtype=torch.float64
    )
    acq = torch.tensor([1.0], dtype=torch.float64)
    return candidate, acq


def test_reproducibility_regression(monkeypatch, tmp_path):
    monkeypatch.setattr("mfbo4bio.bo_core.engine.sampling", _fake_sampling)
    monkeypatch.setattr(
        "mfbo4bio.bo_core.engine.vl.conduct_experiment", _fake_experiment
    )
    monkeypatch.setattr("mfbo4bio.bo_core.engine.train_gp_model", _fake_train_gp_model)
    monkeypatch.setattr(
        "mfbo4bio.bo_core.engine.build_acquisition", _fake_build_acquisition
    )
    monkeypatch.setattr(
        "mfbo4bio.bo_core.engine.propose_batch_for_fidelity",
        _fake_propose_batch_for_fidelity,
    )

    cfg = RunConfig(
        method="qUCB",
        output_name="regression_fixture",
        seed=11,
        results_root=str(tmp_path),
        experiment=ExperimentConfig(clone_distribution="alpha", mbr_level=7),
        bo=BOConfig(n_iterations=2),
    )
    result = run_bo(cfg)

    fixture_path = Path("tests/fixtures/bo_regression_snapshot.json")
    expected = json.loads(fixture_path.read_text())

    actual = {
        "best_values": result.payload["best_values"],
        "fidelities": result.payload["fidelities"],
        "cumulative_cost_list": result.payload["cumulative_cost_list"],
    }
    assert actual == expected
