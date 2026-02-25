from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal

MethodName = Literal["qUCB", "qLogEI", "GIBBON", "industrial"]


@dataclass(slots=True)
class ExperimentConfig:
    clone_distribution: Literal["alpha", "beta"] = "alpha"
    mbr_level: int = 7
    feeding_max: float = 50.0
    temperature_bounds: tuple[float, float] = (30.0, 40.0)
    ph_bounds: tuple[float, float] = (6.0, 8.0)

    def __post_init__(self) -> None:
        if self.mbr_level not in (1, 2, 3, 4, 5, 6, 7, 8, 9):
            raise ValueError("mbr_level must be an integer in [1, 9]")
        if self.feeding_max <= 0:
            raise ValueError("feeding_max must be > 0")
        if self.temperature_bounds[0] >= self.temperature_bounds[1]:
            raise ValueError("temperature_bounds must have low < high")
        if self.ph_bounds[0] >= self.ph_bounds[1]:
            raise ValueError("ph_bounds must have low < high")


@dataclass(slots=True)
class BOConfig:
    n_iterations: int = 10
    task_representation: Literal["HYBRID", "ICM_WRAPPED"] = "ICM_WRAPPED"
    embed_dim: int = 3
    beta: float = 1.0
    batch_size_by_fidelity: Dict[int, int] = field(default_factory=dict)
    cost_by_fidelity: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_iterations < 0:
            raise ValueError("n_iterations must be >= 0")
        if self.embed_dim < 1:
            raise ValueError("embed_dim must be >= 1")
        if self.beta <= 0:
            raise ValueError("beta must be > 0")


@dataclass(slots=True)
class RunConfig:
    method: MethodName
    output_name: str
    seed: int | None = None
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    bo: BOConfig = field(default_factory=BOConfig)
    results_root: str = "results"
    logging_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    reproducible: bool = True

    def __post_init__(self) -> None:
        if not self.output_name:
            raise ValueError("output_name must not be empty")
        if self.results_root.strip() == "":
            raise ValueError("results_root must not be blank")
