import multiprocessing
import os
from itertools import product
from subprocess import run


def run_single_test_job(
    test_type,
    mbr_level,
    clone_distribution,
    iterations,
    seed,
    date,
    task_representation,
):
    """
    Runs a single test job by executing an external Python script.

    This function constructs a file path for the output and,
    if the file does not already exist, it calls `run_single_test.py`
    as a subprocess with a set of command-line arguments.
    This is designed to be used in parallel processing.

    Parameters
    ----------
    test_type : str
        The type of test to run, e.g., "qUCB", "qLogEI".
    mbr_level : int
        The medium-fidelity level for the test.
    clone_distribution : str
        The cell clone distribution to use, e.g., "alpha", "beta".
    iterations : int
        The number of iterations for the test run.
    seed : int
        The random seed for the test run.
    date : str
        A date string to be used in the output file name.
    task_representation : str
        The task representation, e.g., "HYBRID", "ICM_WRAPPED".

    Returns
    -------
    None
        This function does not return a value.
        It executes a subprocess and prints a message if a file already exists.
    """

    output_file = f"./results/bo/{clone_distribution}/{test_type}/{test_type}_{task_representation}_{date}_{seed+1}.json"

    if os.path.exists(output_file):
        print(f"Skipping existing run: {output_file}")
        return

    run(
        [
            "python",
            "run_single_test.py",
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
        ]
    )


def main():
    iterations = 5
    mbr_levels = [7]
    clone_distributions = ["alpha", "beta"]
    test_types = ["qUCB", "qLogEI", "GIBBON"]
    seeds = list(range(10))
    date = ["20250720"]
    task_representation = ["HYBRID", "ICM_WRAPPED"]

    jobs = list(
        product(
            test_types,
            mbr_levels,
            clone_distributions,
            [iterations],
            seeds,
            date,
            task_representation,
        )
    )

    n_cpus = int(os.environ.get("NUM_PROCESSES", multiprocessing.cpu_count()))
    with multiprocessing.Pool(processes=n_cpus) as pool:
        pool.starmap(run_single_test_job, jobs)


if __name__ == "__main__":
    main()
