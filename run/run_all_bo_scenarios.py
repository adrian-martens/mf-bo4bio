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
    # embed_dim,

):
    # embed_suffix = f"_embed{embed_dim}" if task_representation == "HYBRID" else ""
    # output_file = os.path.join(
    #     "results",
    #     "bo",
    #     clone_distribution,
    #     test_type,
    #     f"{test_type}_{task_representation}{embed_suffix}_{date}_{seed+1}.json",
    # )
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
            # "--embed_dim",
            # str(embed_dim),
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
    # jobs = []
    # for task_representation in preset.task_representations:
    #     embed_dims = (
    #         preset.embed_dims if task_representation == "HYBRID" else [preset.embed_dims[0]]
    #     )
    #     jobs.extend(
    #         product(
    #             preset.test_types,
    #             preset.mbr_levels,
    #             preset.clone_distributions,
    #             [preset.iterations],
    #             preset.seeds,
    #             preset.dates,
    #             [task_representation],
    #             embed_dims,
    #         )
    #     )
    n_cpus = int(os.environ.get("NUM_PROCESSES", 4))
    with multiprocessing.Pool(processes=n_cpus) as pool:
        pool.starmap(run_single_test_job, jobs)


if __name__ == "__main__":
    main()
