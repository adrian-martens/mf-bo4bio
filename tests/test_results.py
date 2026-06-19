from pathlib import Path

from mfbo4bio.config import RunConfig
from mfbo4bio.results import bo_output_paths


def test_bo_output_paths_gibbon_compat():
    cfg = RunConfig(method="GIBBON", output_name="abc")
    paths = bo_output_paths(cfg)
    assert len(paths) == 2
    assert any("gibbon_icm" in str(p) for p in paths)
    assert any("/GIBBON/" in str(p).replace("\\\\", "/") for p in paths)


def test_bo_output_paths_single_for_qucb():
    cfg = RunConfig(method="qUCB", output_name="abc")
    paths = bo_output_paths(cfg)
    assert len(paths) == 1
    assert isinstance(paths[0], Path)
