import numpy as np
import torch
from botorch.optim import optimize_acqf
from utils import sampling

from mfbo4bio import conditions_data as data

dtype = torch.float64
feeding_max = data.feeding_max


def custom_optimization(
    X_mean,
    X_std,
    acq_function,
    bounds,
    q=12,
    resolution=3,
    mode="sampling",
    seed=42,
    process_parameters=data.process_parameters_alpha,
):
    """
    Performs a custom optimization routine for an acquisition function.

    Parameters
    ----------
    X_mean : torch.Tensor
        The mean values used for normalizing the input features.
    X_std : torch.Tensor
        The standard deviation values used for normalizing the input features.
    acq_function : botorch.acquisition.AcquisitionFunction
        The acquisition function to be optimized.
    bounds : torch.Tensor
        A 2x2 tensor defining the optimization bounds for the first two dimensions.
    q : int, optional
        The number of candidates in the batch to be optimized. Defaults to 12.
    resolution : int, optional
        The resolution of the initial grid for generating restarts.
        The number of restarts is `resolution**2`.
        Defaults to 3.
    mode : str, optional
        The optimization mode, either "sequential" or "sampling".
        Defaults to "sampling".
    seed : int, optional
        A seed for the random number generator to ensure reproducibility.
        Defaults to 42.
    process_parameters : dict, optional
        A dictionary containing process parameters for different cell types.
        Defaults to `data.process_parameters_alpha`.

    Returns
    -------
    tuple
        A tuple containing:
        - best_candidates (torch.Tensor): The batch of candidates
          that yielded the highest total acquisition value.
        - best_acq_value (torch.Tensor): The corresponding maximum acquisition value.

    Notes
    -----
    - In "sequential" mode, candidates are optimized one at a time,
      with subsequent candidates being conditioned on the previously selected ones.
    - In "sampling" mode, a full batch of candidates is generated via
      a sampling strategy and its total acquisition value is evaluated.
      This is repeated for each restart.
    """
    best_acq_value = None
    best_candidates = None

    num_restarts = resolution**2

    x = np.linspace(bounds[0][0], bounds[1][0], resolution)
    y = np.linspace(bounds[0][1], bounds[1][1], resolution)
    xx, yy = np.meshgrid(x, y)
    grid = np.c_[xx.ravel(), yy.ravel()]

    if mode == "sequential":
        for i in range(num_restarts):
            candidate_list = []
            acq_value_list = []

            shared_0 = grid[i][0]
            shared_1 = grid[i][1]

            fixed_features = {
                0: shared_0.item(),
                1: shared_1.item(),
                2: ((0 - X_mean[2]) / X_std[2]).item(),
                3: ((feeding_max - X_mean[3]) / X_std[3]).item(),
                4: ((0 - X_mean[4]) / X_std[4]).item(),
                5: ((0 - X_mean[5]) / X_std[5]).item(),
            }

            for j in range(q):
                if j > 0:
                    first_candidate = candidate_list[0].detach()
                    fixed_features[0] = first_candidate[0, 0].item()
                    fixed_features[1] = first_candidate[0, 1].item()

                candidate_j, acq_val_j = optimize_acqf(
                    acq_function=acq_function,
                    bounds=bounds,
                    q=1,
                    num_restarts=5,
                    raw_samples=256,
                    fixed_features=fixed_features,
                )

                candidate_list.append(candidate_j)

                if acq_val_j >= 0:
                    acq_value_list.append(acq_val_j)

                X_pending = torch.cat(candidate_list, dim=0)
                acq_function.set_X_pending(X_pending)

            total_acq = torch.stack(acq_value_list).sum()

            if best_acq_value is None or total_acq > best_acq_value:
                best_acq_value = total_acq
                best_candidates = torch.cat(candidate_list, dim=0)

            acq_function.set_X_pending(None)

        return best_candidates, best_acq_value

    if mode == "sampling":
        dimensions_dict = {
            "fidelity": ("discrete", [0, 7, 10]),
            "clone": ("discrete", list(range(len(process_parameters.keys())))),
        }

        fidelity_distribution = {0: 1, 7: 0, 10: 0}

        for i in range(num_restarts):
            candidate_list = []
            acq_value_list = []

            shared_0 = grid[i][0]
            shared_1 = grid[i][1]

            row = torch.tensor(
                [
                    shared_0,
                    shared_1,
                    (0 - X_mean[2]) / X_std[2],
                    (feeding_max - X_mean[3]) / X_std[3],
                    (0 - X_mean[4]) / X_std[4],
                    (0 - X_mean[5]) / X_std[5],
                ],
                dtype=torch.float64,
            )

            candidate_i = row.repeat(12, 1)

            candidate_clones = torch.tensor(
                sampling(
                    dimensions_dict=dimensions_dict,
                    num_samples=12,
                    seed=seed,
                    fidelity_distribution=fidelity_distribution,
                ),
                dtype=torch.float64,
            )[:, 1].unsqueeze(-1)

            candidate_i = torch.cat((candidate_i, candidate_clones), 1)

            for j in range(candidate_i.shape[0]):
                X_j = candidate_i[j].unsqueeze(0)
                X_pending = candidate_i[: j + 1] if j > 0 else None

                acq_function.set_X_pending(X_pending)
                acq_val_j = acq_function(X_j)
                acq_value_list.append(max(0, acq_val_j.item()))

            total_acq = torch.tensor(sum(acq_value_list))

            if best_acq_value is None or total_acq > best_acq_value:
                best_acq_value = total_acq
                best_candidates = candidate_i

            acq_function.set_X_pending(None)

        return best_candidates, best_acq_value
