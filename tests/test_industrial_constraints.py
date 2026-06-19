import numpy as np

from mfbo4bio.industrial_methods import apply_feeding_constraints


def test_apply_feeding_constraints_low_fidelity():
    dims = ["temperature", "feeding1", "feeding2", "feeding3"]
    samples = np.array([[35.0, 1.0, 2.0, 3.0]])
    out = apply_feeding_constraints(
        samples.copy(),
        dims,
        fidelity_value=0,
        constraints={
            "feeding_max": 50,
            "feeding_dims": ["feeding1", "feeding2", "feeding3"],
        },
    )
    assert out[0, 1] == 0
    assert out[0, 2] == 50
    assert out[0, 3] == 0
