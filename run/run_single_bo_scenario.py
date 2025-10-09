import argparse

import mfbo4bio.mfbo_GIBBON as GIBBON
import mfbo4bio.mfbo_qLogEI as qLogEI
import mfbo4bio.mfbo_qUCB as qUCB


def main():
    """
    Parses command-line arguments and runs a multi-fidelity Bayesian optimization test.

    This script serves as an entry point to run different multi-fidelity
    optimization algorithms (GIBBON, qLogEI, qUCB) by taking various
    test parameters from the command line. It dynamically imports and
    calls the appropriate `run` function based on the `--test_type` argument.

    Command-line Arguments
    ----------------------
    --test_type : str, required
        The name of the test to run. Must be one of "GIBBON", "qUCB", or "qLogEI".
    --iterations : int, required
        The number of iterations for the optimization algorithm.
    --seed : int, required
        The random seed for the test run.
    --mbr_level : float, required
        The medium-fidelity level.
    --clone_distribution : str, required
        The cell clone distribution to use, e.g., "alpha" or "beta".
    --date : str, required
        A date string to be used in the output file name.
    --task_representation : str, required
        The task representation, e.g., "HYBRID" or "ICM_WRAPPED".

    Raises
    ------
    SystemExit
        If required command-line arguments are not provided.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_type", required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mbr_level", type=float, required=True)
    parser.add_argument("--clone_distribution", type=str, required=True)
    parser.add_argument("--date", type=str, required=True)
    parser.add_argument("--task_representation", type=str, required=True)

    args = parser.parse_args()

    output_name = (
        f"{args.test_type}_{args.task_representation}_{args.date}_{args.seed + 1}"
    )

    if args.test_type == "GIBBON":
        GIBBON.run(
            args.iterations,
            output_name=output_name,
            seed=args.seed,
            clone_distribution=args.clone_distribution,
            mbr_level=args.mbr_level,
            task_representation=args.task_representation,
        )
    elif args.test_type == "qUCB":
        qUCB.run(
            args.iterations,
            output_name=output_name,
            seed=args.seed,
            clone_distribution=args.clone_distribution,
            mbr_level=args.mbr_level,
            task_representation=args.task_representation,
        )
    elif args.test_type == "qLogEI":
        qLogEI.run(
            args.iterations,
            output_name=output_name,
            seed=args.seed,
            clone_distribution=args.clone_distribution,
            mbr_level=args.mbr_level,
            task_representation=args.task_representation,
        )


if __name__ == "__main__":
    main()
