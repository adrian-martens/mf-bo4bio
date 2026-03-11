import argparse

from mfbo4bio.config import BOConfig, ExperimentConfig, RunConfig
from mfbo4bio.pipeline import run_bo


def run(
    n_iterations=10,
    output_name="gibbon_results",
    seed=None,
    clone_distribution="alpha",
    mbr_level=7,
    task_representation="HYBRID",
    embed_dim=3,
):
    config = RunConfig(
        method="GIBBON",
        output_name=output_name,
        seed=seed,
        experiment=ExperimentConfig(
            clone_distribution=clone_distribution,
            mbr_level=int(mbr_level),
        ),
        bo=BOConfig(
            n_iterations=n_iterations,
            task_representation=task_representation,
            embed_dim=embed_dim,
        ),
    )
    run_bo(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Bayesian Optimization with configurable parameters."
    )
    parser.add_argument(
        "--n_iterations", type=int, default=10, help="Number of BO iterations"
    )
    parser.add_argument(
        "--json_output", type=str, default="gibbon_results", help="Output JSON file"
    )

    args = parser.parse_args()
    run(
        n_iterations=args.n_iterations,
        output_name=args.json_output,
    )
