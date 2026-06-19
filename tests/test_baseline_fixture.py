import json
from pathlib import Path


def test_seeded_short_runs_fixture_complete():
    data = json.loads(Path("tests/fixtures/seeded_short_runs.json").read_text())
    assert set(data.keys()) == {"qUCB", "qLogEI", "GIBBON"}
    for method in data.values():
        assert set(method.keys()) == {"alpha", "beta"}
        for val in method.values():
            assert "seed" in val and "n_iterations" in val
