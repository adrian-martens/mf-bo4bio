import multiprocessing
import os
from itertools import product
from subprocess import run

from mfbo4bio.presets import DEFAULT_BO_PRESET


def run_single_test_job(
    test_type,
    mbr_level,
    clone_distribution,
    iterations,
    seed,
    date,
    task_representation,
):
    output_file = os.path.join(
        "results",
        "bo",
        clone_distribution,
        test_type,
        f"{test_type}_{task_representation}_{date}_{seed+1}.json",
    )

    if os.path.exists(output_file):
        print(f"Skipping existing run: {output_file}")
        return

    run(
        [
            "python",
            "./run/run_single_bo_scenario.py",
            "--test_type",
            test_type,
            "--iterations",
            str(iterations),
            "--seed",
            str(seed),
            "--mbr_level",
            str(mbr_level),
            "--clone_distribution",
            clone_distribution,
            "--date",
            date,
            "--task_representation",
            task_representation,
        ],
        check=True,
    )


def main():
    preset = DEFAULT_BO_PRESET
    jobs = list(
        product(
            preset.test_types,
            preset.mbr_levels,
            preset.clone_distributions,
            [preset.iterations],
            preset.seeds,
            preset.dates,
            preset.task_representations,
        )
    )

    n_cpus = int(os.environ.get("NUM_PROCESSES", 4))
    with multiprocessing.Pool(processes=n_cpus) as pool:
        pool.starmap(run_single_test_job, jobs)


if __name__ == "__main__":
    main()
