from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BOScenarioPreset:
    iterations: int
    mbr_levels: list[int]
    clone_distributions: list[str]
    test_types: list[str]
    seeds: list[int]
    dates: list[str]
    task_representations: list[str]
    # embed_dims: list[int]


DEFAULT_BO_PRESET = BOScenarioPreset(
    iterations=50,
    mbr_levels=[7],
    clone_distributions=["alpha", "beta"],
    test_types=["GIBBON", "qLogEI", "qUCB"],
    seeds=list(range(10)),
    dates=["DIM_TEST"],
    task_representations=["HYBRID", "ICM_WRAPPED"],
    # embed_dims=[2, 3, 5, 6, 10],
)


@dataclass(frozen=True, slots=True)
class IndustrialPreset:
    clone_dist: str
    platform_cond: bool
    date: str
    repeats: int


DEFAULT_INDUSTRIAL_PRESET = IndustrialPreset(
    clone_dist="alpha",
    platform_cond=True,
    date="TEST",
    repeats=15,
)
