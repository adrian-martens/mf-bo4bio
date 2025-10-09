import pprint
import random

import numpy as np


def generate_process_parameters(
    n_celltypes=5, seed=None, filename="cell_parameters.py"
):
    """
    Generates a dictionary of random cell culture process parameters
    for different cell types.

    Parameters
    ----------
    n_celltypes : int, optional
        The number of unique cell types to generate parameters for. Defaults to 5.
    seed : int, optional
        A seed for the random number generators to ensure reproducibility
        of the generated parameters. Defaults to None.
    filename : str, optional
        The name of the Python file to which the generated parameters will be saved.
        The file will be formatted as a dictionary. Defaults to "cell_parameters.py".

    Returns
    -------
    dict
        A dictionary containing the generated process parameters.
        Each key is a string like 'celltype_X', and the value is
        a dictionary of parameters for that cell type.

    Raises
    ------
    IOError
        If there is an issue writing to the specified `filename`,
        e.g., due to a permission error.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    def rand_float(low, high):
        return round(np.random.uniform(low, high), 10)

    process_parameters = {}

    for i in range(1, n_celltypes + 1):
        celltype = f"celltype_{i}"
        process_parameters[celltype] = {
            "my_max": rand_float(0.02, 0.06),
            "K_lysis": rand_float(0.03, 0.05),
            "k": [
                rand_float(1e-3, 1.5e-3),
                rand_float(1e-2, 1.5e-2),
                rand_float(1e-2, 1.5e-2),
            ],
            "K": [
                rand_float(130, 170),
                rand_float(35, 40),
                rand_float(0.9, 1.6),
                rand_float(0.21, 0.3),
            ],
            "Y": [
                rand_float(1e7, 3e8),
                rand_float(1e8, 1e10),
                rand_float(1, 3),
                rand_float(0.6, 1.0),
                rand_float(1e-8, 2e-7),
                rand_float(3e-6, 8e-6),
            ],
            "m": [
                rand_float(1e-13, 5e-10),
                rand_float(3e-12, 5e-12),
            ],
            "A": rand_float(2, 5),
            "pH_opt": rand_float(6.0, 8.0),
            "E_a": rand_float(30, 40),
        }

    # Pretty-print the dictionary into the .py file
    with open(filename, "w") as f:
        f.write("process_parameters = \\\n")
        pprint.pprint(
            process_parameters, stream=f, indent=4, width=100, sort_dicts=False
        )

    print(f"Saved nicely formatted process_parameters to '{filename}'")
    return process_parameters


generate_process_parameters(100, 30)  # 42
