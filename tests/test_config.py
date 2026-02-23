from mfbo4bio.config import BOConfig, ExperimentConfig, RunConfig


def test_config_valid_defaults():
    cfg = RunConfig(method="qUCB", output_name="x")
    assert cfg.output_name == "x"
    assert cfg.bo.n_iterations == 10


def test_config_validation():
    try:
        ExperimentConfig(mbr_level=11)
        assert False, "Expected ValueError"
    except ValueError:
        pass

    try:
        BOConfig(beta=0)
        assert False, "Expected ValueError"
    except ValueError:
        pass
