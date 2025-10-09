import argparse
import json
import os
import traceback

import numpy as np
import torch
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.max_value_entropy_search import (
    qMultiFidelityLowerBoundMaxValueEntropy,
)
from botorch.models.deterministic import DeterministicModel
from botorch.optim import optimize_acqf
from tqdm import tqdm

import mfbo4bio.conditions_data as data
import mfbo4bio.virtual_lab as vl
from mfbo4bio.optimization import custom_optimization
from mfbo4bio.utils import (
    destandardize_mixed_tensor,
    sampling,
    standardize_data,
    standardize_mixed_tensor,
    train_gp_model,
)


class FixedFidelityCostModel(DeterministicModel):
    """
    Initializes a simple fixed fidelity cost model, which can be used for
    multi-fidelity optimization. In botorch implementations this is a way
    of making optimizations cost-aware.

    Parameters
    ----------
    cost_map : dict[float, float]
        A dictionary mapping each fidelity value to its corresponding cost.
        For example, `{0.0: 2.0, 7.0: 10.0, 10.0: 200.0}`.
    fidelity_dim : int, optional
        The index of the fidelity dimension in the input tensor `X`.
        It is assumed to be -2 by default.
    """

    def __init__(
        self,
        cost_map: dict[float, float],
        fidelity_dim: int = -2,
    ) -> None:
        super().__init__()
        self.fidelity_dim = fidelity_dim
        self._num_outputs = 1

        self.fidelity_values = sorted(cost_map)
        self.cost_tensor = torch.tensor(
            [cost_map[f] for f in self.fidelity_values], dtype=torch.float
        )
        self.register_buffer("costs", self.cost_tensor)
        self.register_buffer("fidelities", torch.tensor(self.fidelity_values))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Evaluate cost based on fidelity values in X."""
        fidelity_vals = X[..., self.fidelity_dim]
        cost = torch.zeros_like(fidelity_vals)

        for i, fval in enumerate(self.fidelities):
            cost = torch.where(fidelity_vals == fval, self.costs[i], cost)

        return cost.unsqueeze(-1)

    @property
    def num_outputs(self) -> int:
        return self._num_outputs


def run(
    n_iterations=10,
    output_name="gibbon_results",
    seed=None,
    clone_distribution="alpha",
    mbr_level=7,
    task_representation="ICM_WRAPPED",
    embed_dim=None,
):
    """
    Runs a multi-fidelity Bayesian optimization simulation
    using the GIBBON acquisition function.

    Parameters
    ----------
    n_iterations : int, optional
        The number of Bayesian optimization iterations to perform. Defaults to 10.
    output_name : str, optional
        The base name for the output JSON file. Defaults to "gibbon_results".
    seed : int, optional
        The random seed for reproducibility. Defaults to None.
    clone_distribution : str, optional
        The type of cell clone distribution to use, either "alpha" or "beta".
        Defaults to "alpha".
    mbr_level : int, optional
        The medium-fidelity level to be used in the simulation. Defaults to 7.
    task_representation : str, optional
        The representation of the task for the GP model.
        Can be "ICM_WRAPPED" or "HYBRID". Defaults to "ICM_WRAPPED".
    embed_dim : int, optional
        The dimension of the latent space for the task representation,
        used only for "HYBRID" mode. Defaults to None.

    Returns
    -------
    None
        The function saves the results to a JSON file and does not return any value.

    Raises
    ------
    NameError
        If an unsupported `clone_distribution` is specified.
    Exception
        Catches and reports any other exceptions that occur during the simulation.
    """
    if clone_distribution == "alpha":
        process_parameters = data.process_parameters_alpha
    elif clone_distribution == "beta":
        process_parameters = data.process_parameters_beta
    else:
        raise NameError(
            f"No clone distribution named {clone_distribution}. \
            Please specify 'alpha' or 'beta' as input"
        )

    error_message = None

    try:
        rng = np.random.default_rng(seed)

        T = rng.uniform(30, 40)
        pH = rng.uniform(6, 8)
        dtype = torch.float64
        feeding_max = 50
        dimensions_dict = {
            "temperature": ("continuous", (T - 0.0001, T)),
            "ph": ("continuous", (pH - 0.0001, pH)),
            "feeding1": ("continuous", (0, feeding_max)),
            "feeding2": ("continuous", (0, feeding_max)),
            "feeding3": ("continuous", (0, feeding_max)),
            "fidelity": ("discrete", [0, mbr_level, 10]),
            "clone": ("discrete", list(range(len(process_parameters.keys())))),
        }

        constraints = {
            "feeding_max": feeding_max,
            "feeding_dims": ["feeding1", "feeding2", "feeding3"],
        }

        fidelity_distribution = {0: 0.9, mbr_level: 0.1, 10: 0}

        X_initial = sampling(
            method="latin_hypercube",
            dimensions_dict=dimensions_dict,
            num_samples=12,
            constraints=constraints,
            fidelity_distribution=fidelity_distribution,
            seed=rng,
        )

        # Evaluate at initial datapoints
        y_initial = vl.conduct_experiment(
            X_initial,
            clone_distribution=clone_distribution,
            mbr_level=mbr_level,
            rng=rng,
        )

        best_initial = X_initial[np.argmax(y_initial)]

        X_continuous = X_initial[:, :-1]
        X_categorical = X_initial[:, -1].reshape(-1, 1)

        X_standardized, y_standardized, X_mean, X_std, y_mean, y_std = standardize_data(
            X_continuous, y_initial
        )

        Xtrain = torch.tensor(np.hstack([X_standardized, X_categorical]))
        y_initial = torch.as_tensor(y_initial, dtype=dtype).unsqueeze(-1)

        ytrain = torch.tensor(y_standardized, dtype=dtype).unsqueeze(-1)

        # Saving the best Values
        best_values = []
        best_from_batch = []
        batches = []
        fidelities = [0]
        best_values.append(ytrain.max().item())
        best_from_batch.append(best_initial)
        batches.append(X_initial)

        cumulative_cost = 0
        cumulative_cost_list = [0]
        fidelity_level = [0, mbr_level, 10]
        cost_level = {0: 10, mbr_level: 575, 10: 2100}

        cumulative_cost += cost_level[0] * 12
        cumulative_cost_list.append(cumulative_cost)

        # GP Training
        if task_representation == "HYBRID":
            model, likelihood = train_gp_model(
                Xtrain, ytrain, model_type=task_representation, embed_dim=embed_dim
            )
        else:
            model, likelihood = train_gp_model(
                Xtrain, ytrain, model_type=task_representation
            )

        # ============= Bayesian Optimization Loop ============ #
        for iteration in tqdm(range(n_iterations)):

            print(
                f"=========== BO ITERATION {iteration+1}/{n_iterations} =========== \n"
            )

            print("GP-Paramenters \n")
            if model.covar_module.lengthscale is not None:
                print("Lengthscale:", model.covar_module.lengthscale.detach().numpy())
            else:
                print("Lengthscale not learned yet.")
            print("\n")

            batch_size = {0: 12, mbr_level: 4, 10: 1}

            candidate_set = sampling(
                method="latin_hypercube",
                dimensions_dict=dimensions_dict,
                num_samples=512,
                constraints=constraints,
                fidelity_distribution={0: 0, mbr_level: 0, 10: 1},
                seed=rng,
            )
            candidate_set_continuous = candidate_set[:, :-1]
            candidate_set_categorical = torch.tensor(
                candidate_set[:, -1], dtype=dtype
            ).unsqueeze(-1)
            candidate_set_standardized = (
                torch.tensor(candidate_set_continuous, dtype=dtype) - X_mean
            ) / X_std
            candidate_set_standardized = torch.hstack(
                (candidate_set_standardized, candidate_set_categorical)
            )

            lower_bounds = torch.tensor(
                (np.array([30.0, 6.0, 0.0, 0.0, 0.0, 0]) - X_mean) / X_std, dtype=dtype
            )
            lower_bounds = torch.hstack((lower_bounds, torch.tensor(0)))
            upper_bounds = torch.tensor(
                (np.array([40.0, 8.0, 50.0, 50.0, 50.0, 10]) - X_mean) / X_std,
                dtype=dtype,
            )
            upper_bounds = torch.hstack(
                (upper_bounds, torch.tensor(len(process_parameters.keys()) - 1))
            )

            gibbon_values = {0: 0, mbr_level: 0, 10: 0}
            batches_standardized = {0: [], mbr_level: [], 10: []}

            def project_to_fidelity(X):
                X = X.clone()
                X[..., -2] = (10 - X_mean[-1]) / X_std[-1]
                return X

            gibbon_acq = qMultiFidelityLowerBoundMaxValueEntropy(
                model=model,
                candidate_set=candidate_set_standardized,
                project=project_to_fidelity,
                use_gumbel=True,
                cost_aware_utility=InverseCostWeightedUtility(
                    cost_model=FixedFidelityCostModel(cost_level)
                ),
            )

            for i in fidelity_level:

                print(f"Fidelity level: {i}")

                bounds = torch.stack([lower_bounds, upper_bounds])

                fixed_features = {
                    5: torch.tensor((i - X_mean[5]) / X_std[5], dtype=dtype)
                }

                # Standardized sum values
                mean_sum = X_mean[2:5].sum()
                std_sum = np.sqrt(X_std[2] ** 2 + X_std[3] ** 2 + X_std[4] ** 2)

                # Standardize feeding_max
                standardized_feeding_max = (feeding_max - mean_sum) / std_sum

                if i == 0:
                    print("Start Custom Optimization")
                    candidate_standardized, acq_value = custom_optimization(
                        X_mean,
                        X_std,
                        gibbon_acq,
                        bounds,
                        batch_size[i],
                        resolution=5,
                        mode="sampling",
                        seed=int(rng.integers(0, 2**32 - 1)),
                    )

                else:

                    tol = 0.4

                    inequality_constraints = [
                        # Lower bound:
                        # x[2] + x[3] + x[4] >= standardized_feeding_max - tol
                        (
                            torch.tensor([2, 3, 4], dtype=torch.long),
                            torch.tensor([1.0, 1.0, 1.0], dtype=dtype),
                            standardized_feeding_max - tol,
                        ),
                        # Upper bound:
                        # x[2] + x[3] + x[4] <= standardized_feeding_max
                        (
                            torch.tensor([2, 3, 4], dtype=torch.long),
                            torch.tensor([-1.0, -1.0, -1.0], dtype=dtype),
                            -standardized_feeding_max,
                        ),
                    ]

                    candidate_standardized, acq_value = optimize_acqf(
                        acq_function=gibbon_acq,
                        bounds=bounds,
                        q=batch_size[i],
                        num_restarts=5,
                        raw_samples=512,
                        sequential=True,
                        fixed_features=fixed_features,
                        inequality_constraints=inequality_constraints,
                    )

                candidate_standardized.detach().numpy()

                print(
                    "Candidates:",
                    destandardize_mixed_tensor(candidate_standardized, X_mean, X_std),
                )  # Convert back to original scale

                gibbon_values[i] = ((acq_value) / (batch_size[i] * cost_level[i])).sum()

                batches_standardized[i] = candidate_standardized

            selected_fidelity = max(gibbon_values, key=gibbon_values.get)

            if iteration == n_iterations - 1:
                selected_fidelity = 10

            print(f"selected fidelity for iteration {iteration+1}: {selected_fidelity}")

            # Save information about selected batch and add cost
            cumulative_cost += (
                batch_size[selected_fidelity] * cost_level[selected_fidelity]
            )
            cumulative_cost_list.append(cumulative_cost)
            fidelities.append(selected_fidelity)

            # Destandardize selected batch
            batch = destandardize_mixed_tensor(
                batches_standardized[selected_fidelity], X_mean, X_std
            )
            batch[:, -1] = torch.round(batch[:, -1])

            print(
                f"Selected Batch on Fidelity {selected_fidelity}",
                "\n",
                np.round(batch, decimals=2),
                "\n",
            )

            # Evaluate batch
            new_y = torch.tensor(
                vl.conduct_experiment(
                    batch,
                    clone_distribution=clone_distribution,
                    mbr_level=mbr_level,
                    rng=rng,
                ),
                dtype=dtype,
            ).unsqueeze(-1)

            best_from_batch.append(batch[np.argmax(np.array(new_y))])
            batches.append(batch)

            # Update training data with batch and correspondong target values
            X_initial = np.vstack([X_initial, batch])
            y_initial = np.concatenate([y_initial, new_y])

            # Save best target value so far
            print(
                f"Best Value so far: {float(y_initial.max())} \
                with X: {X_initial[np.argmax(y_initial)]} \n"
            )
            best_values.append(float(y_initial.max()))

            X_initial_continuous = X_initial[:, :-1]

            X_standardized, y_standardized, X_mean, X_std, y_mean, y_std = (
                standardize_data(X_initial_continuous, y_initial)
            )

            X_standardized = standardize_mixed_tensor(
                torch.tensor(X_initial), X_mean, X_std
            )

            Xtrain = torch.tensor(X_standardized, dtype=dtype).clone().detach()
            ytrain = torch.tensor(y_standardized, dtype=dtype)

            # Retrain GP
            if task_representation == "HYBRID":
                model, likelihood = train_gp_model(
                    Xtrain, ytrain, model_type=task_representation, embed_dim=embed_dim
                )
            else:
                model, likelihood = train_gp_model(
                    Xtrain, ytrain, model_type=task_representation
                )

    except Exception as e:
        print(f"Error encountered: {e}")
        traceback.print_exc()
        error_message = str(e)
    finally:
        bo_results = {
            "clone distribution": clone_distribution,
            "mbr_level": mbr_level,
            "n_iterations": batch_size,
            "iterations": n_iterations,
            "best_values": best_values,
            "batches": [
                (
                    batch.tolist()
                    if isinstance(batch, (np.ndarray, torch.Tensor))
                    else batch
                )
                for batch in batches
            ],
            "best_points": [
                (
                    point.tolist()
                    if isinstance(point, (np.ndarray, torch.Tensor))
                    else point
                )
                for point in best_from_batch
            ],
            "cumulative_cost_list": cumulative_cost_list,
            "fidelities": fidelities,
            "Xtrain": Xtrain.tolist() if isinstance(Xtrain, torch.Tensor) else Xtrain,
            "ytrain": ytrain.tolist() if isinstance(ytrain, torch.Tensor) else ytrain,
            "X_mean": (
                X_mean.tolist()
                if isinstance(X_mean, (np.ndarray, torch.Tensor))
                else X_mean
            ),
            "X_std": (
                X_std.tolist()
                if isinstance(X_std, (np.ndarray, torch.Tensor))
                else X_std
            ),
            "y_mean": (
                float(y_mean)
                if isinstance(y_mean, (np.ndarray, torch.Tensor))
                else y_mean
            ),
            "y_std": (
                float(y_std) if isinstance(y_std, (np.ndarray, torch.Tensor)) else y_std
            ),
            "error": error_message,
        }

        os.makedirs(
            f"../results/bo/{clone_distribution}/gibbon_icm",
            exist_ok=True,
        )
        # Save structured data as JSON
        with open(
            f"../results/bo/{clone_distribution}/gibbon_icm_{output_name}.json",
            "w",
        ) as f:
            json.dump(bo_results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Bayesian Optimization with configurable parameters."
    )
    parser.add_argument(
        "--n_iterations", type=int, default=10, help="Number of BO iterations"
    )
    parser.add_argument(
        "--batch_size", type=int, default=5, help="Batch size per iteration"
    )
    parser.add_argument(
        "--json_output", type=str, default="gibbon_results", help="Output JSON file"
    )

    args = parser.parse_args()
    run(
        n_iterations=args.n_iterations,
        batch_size=args.batch_size,
        output_name=args.json_output,
    )
