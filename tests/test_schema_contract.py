from mfbo4bio.config import RunConfig
from mfbo4bio.pipeline import run_bo


def test_bo_payload_schema_keys(monkeypatch, tmp_path):
    class DummyResult:
        payload = {
            "clone distribution": "alpha",
            "mbr_level": 7,
            "n_iterations": {0: 12, 7: 4, 10: 1},
            "iterations": 0,
            "best_values": [],
            "batches": [],
            "best_points": [],
            "cumulative_cost_list": [],
            "fidelities": [],
            "Xtrain": [],
            "ytrain": [],
            "X_mean": [],
            "X_std": [],
            "y_mean": 0,
            "y_std": 1,
            "error": None,
            "config": {},
        }

    monkeypatch.setattr("mfbo4bio.pipeline.run_bo_engine", lambda cfg: DummyResult())
    result = run_bo(
        RunConfig(method="qUCB", output_name="schema", results_root=str(tmp_path))
    )
    required = {
        "clone distribution",
        "mbr_level",
        "n_iterations",
        "iterations",
        "best_values",
        "batches",
        "best_points",
        "cumulative_cost_list",
        "fidelities",
        "Xtrain",
        "ytrain",
        "X_mean",
        "X_std",
        "y_mean",
        "y_std",
        "error",
    }
    assert required.issubset(result.payload.keys())
