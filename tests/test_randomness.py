from mfbo4bio.randomness import SeedManager


def test_seed_manager_is_deterministic():
    a = SeedManager(123)
    b = SeedManager(123)
    assert a.sobol_seed("x") == b.sobol_seed("x")
    ra = a.numpy_rng("n").normal(size=3)
    rb = b.numpy_rng("n").normal(size=3)
    assert (ra == rb).all()
