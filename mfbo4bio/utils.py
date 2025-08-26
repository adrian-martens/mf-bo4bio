import gpytorch
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.multitask import MultiTaskGP
from botorch.posteriors import GPyTorchPosterior
from gpytorch.constraints import Interval
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from scipy.stats import qmc

import mfbo4bio.conditions_data as data


class TaskFilteredMultiTaskGP(MultiTaskGP):
    @property
    def num_outputs(self) -> int:
        return 1


class EntityEmbeddingKernel(Kernel):
    """
    Kernel that uses an embedding matrix on a single categorical input.

    This kernel computes a covariance matrix based on the Euclidean distance
    between learned embeddings for each categorical entity. It assumes the
    categorical feature is the last column of the input tensors.

    Parameters
    ----------
    num_categories : int
        The number of unique categorical values.
    embed_dim : int
        The dimensionality of the embedding space.
    """

    has_lengthscale = False

    def __init__(self, num_categories: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)

        self.embedding = torch.nn.Parameter(torch.randn(num_categories, embed_dim))
        self.num_categories = num_categories

    def forward(self, x1, x2, diag=False, **params):
        # Get the category indices (assumes last column is categorical)
        x1_cat = x1[..., -1].long()
        x2_cat = x2[..., -1].long()

        # Look up embeddings
        x1_embed = self.embedding[x1_cat]
        x2_embed = self.embedding[x2_cat]

        # Compute distances
        dists = self.covar_dist(
            x1_embed, x2_embed, diag=diag, square_dist=True, **params
        )
        return torch.exp(-0.5 * dists)


class CustomMultiTaskGP(SingleTaskGP):
    """
    A custom Multi-Task GP model for hybrid data.

    This model handles both continuous and categorical features. The covariance
    is a product of a standard RBF kernel for continuous dimensions and a
    learned entity embedding kernel for the categorical dimension.

    Parameters
    ----------
    train_X : torch.Tensor
        The training input tensor.
    train_Y : torch.Tensor
        The training output tensor.
    """

    def __init__(self, train_X, train_Y):

        super().__init__(
            train_X,
            train_Y,
        )

        self.mean_module = ConstantMean()

        dims = train_X.shape[-1]
        self.covar_module = ScaleKernel(
            RBFKernel(ard_num_dims=dims - 1, lengthscale_constraint=Interval(0.01, 5.0))
        )

        self.task_module = EntityEmbeddingKernel(
            num_categories=len(data.process_parameters_beta.keys()), embed_dim=4
        )
        self.add_module("task_module", self.task_module)

    def forward(self, x):
        """
        Defines the forward pass of the GP model.

        Parameters
        ----------
        x : torch.Tensor
            The input data tensor.

        Returns
        -------
        gpytorch.distributions.MultivariateNormal
            The GP posterior distribution.
        """
        x_real = x[..., :-1]
        x_cat = x

        mean_x = self.mean_module(x_real)
        covar_real = self.covar_module(x_real)
        covar_cat = self.task_module(x_cat)

        covar_x = covar_real * covar_cat
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    @property
    def batch_shape(self):
        return torch.Size()

    def posterior(
        self, X, output_indices=None, observation_noise=False, posterior_transform=None
    ):
        """
        Computes the posterior distribution of the model.

        Parameters
        ----------
        X : torch.Tensor
            The input data for which to compute the posterior.
        output_indices : list, optional
            A list of output indices to filter.
        observation_noise : bool, optional
            Whether to include observation noise. Defaults to False.
        posterior_transform : object, optional
            An optional posterior transform to apply.

        Returns
        -------
        botorch.posteriors.GPyTorchPosterior
            The posterior distribution.
        """
        self.eval()
        with torch.no_grad():
            mvn = self.likelihood(self(X)) if observation_noise else self(X)
        return GPyTorchPosterior(mvn)


# Data Standardization
def standardize_data(X, y=None):
    """
    Standardizes input X and output y (if provided) to have a mean of 0 and a
    standard deviation of 1.

    Parameters
    ----------
    X : np.ndarray
        The input data to standardize.
    y : np.ndarray, optional
        The output data to standardize. Defaults to None.

    Returns
    -------
    tuple
        If `y` is provided, returns a tuple containing:
        - X_standardized : The standardized input data.
        - y_standardized : The standardized output data.
        - X_mean : The mean of the original input data.
        - X_std : The standard deviation of the original input data.
        - y_mean : The mean of the original output data.
        - y_std : The standard deviation of the original output data.

        If `y` is not provided, returns a tuple containing:
        - X_standardized : The standardized input data.
        - X_mean : The mean of the original input data.
        - X_std : The standard deviation of the original input data.
    """
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X_standardized = (X - X_mean) / X_std

    if y is not None:
        y = np.array(y)
        y_mean, y_std = y.mean(), y.std()
        y_standardized = (y - y_mean) / y_std
        return X_standardized, y_standardized, X_mean, X_std, y_mean, y_std
    return X_standardized, X_mean, X_std


def unstandardize_y(y_standardized, y_mean, y_std):
    """
    Unstandardizes predictions back to their original scale.

    Parameters
    ----------
    y_standardized : np.ndarray or torch.Tensor
        The standardized output values.
    y_mean : float
        The mean of the original output data.
    y_std : float
        The standard deviation of the original output data.

    Returns
    -------
    np.ndarray or torch.Tensor
        The unstandardized output values.
    """
    return y_standardized * y_std + y_mean


# Train GP Model
def train_gp_model(
    train_x,
    train_y,
    learning_rate=0.005,
    training_iter=5000,
    dims=2,
    model_type="HYBRID",
):
    """
    Trains a GP model using standardized data.

    Parameters
    ----------
    train_x : torch.Tensor
        The input training data.
    train_y : torch.Tensor
        The output training data.
    learning_rate : float, optional
        The learning rate for the optimizer. Note that `fit_gpytorch_mll`
        overrides this. Defaults to 0.005.
    training_iter : int, optional
        The number of training iterations. Note that `fit_gpytorch_mll`
        overrides this. Defaults to 5000.
    dims : int, optional
        The embedding dimension for the categorical features, used by the
        "HYBRID" model. Defaults to 2.
    model_type : str, optional
        The type of GP model to train. Can be "HYBRID" or "ICM_WRAPPED".
        Defaults to "HYBRID".

    Returns
    -------
    tuple
        A tuple containing the trained GP model and its likelihood object.

    Raises
    ------
    ValueError
        If an unsupported `model_type` is provided.
    """

    if model_type == "HYBRID":
        from botorch.fit import fit_gpytorch_mll

        model = CustomMultiTaskGP(train_x, train_y)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll, options={"maxiter": 800})
        return mll.model, mll.likelihood

    elif model_type == "ICM_WRAPPED":
        from botorch.fit import fit_gpytorch_mll

        all_tasks = [j for j in range(len(data.process_parameters_beta.keys()))]
        base_model = MultiTaskGP(
            train_X=train_x,
            train_Y=train_y,
            task_feature=-1,
            all_tasks=all_tasks,
            outcome_transform=None,
        )
        base_model.covar_module.ard_num_dims = train_x.size(-1) - 1

        # Wrap the base model
        model = TaskFilteredMultiTaskGP(
            train_X=train_x,
            train_Y=train_y,
            task_feature=-1,
            all_tasks=all_tasks,
            outcome_transform=None,
        )
        model.load_state_dict(base_model.state_dict())  # Copy trained weights

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll=mll, optimizer_kwargs={"options": {"maxiter": 1000}})
        return mll.model, mll.likelihood

    else:
        raise ValueError("Choose the right model type")


def sampling(
    method="latin_hypercube",
    dimensions_dict={},
    num_samples=10,
    constraints=None,
    seed=None,
    fidelity_distribution=None,
):
    """
    Generate samples using a specified sampling method in a mixed-dimensional space.

    This function supports "latin_hypercube" sampling and can apply constraints
    and control the distribution of fidelity levels in the generated samples.

    Parameters
    ----------
    method : str, optional
        Sampling method. Currently supports "latin_hypercube".
        Defaults to "latin_hypercube".
    dimensions_dict : dict
        A dictionary where keys are dimension names and values are tuples
        specifying type ("continuous" or "discrete") and the range or values.
    num_samples : int, optional
        The total number of samples to generate. Defaults to 10.
    constraints : dict, optional
        A dictionary of constraints, e.g.,
        {"feeding_max": 50, "feeding_dims": ["dim3", "dim4", "dim5"]}.
        Defaults to None.
    seed : int, optional
        A random seed for reproducibility. Defaults to None.
    fidelity_distribution : dict, optional
        A dictionary specifying proportions for
        `fidelity` values, e.g., {0: 0.5, 7: 0.2, 10: 0.3}.
        This is used to ensure a specific number of samples at each fidelity level.
        Defaults to None.

    Returns
    -------
    np.ndarray
        An array of valid samples with the specified characteristics.

    Raises
    ------
    ValueError
        If an unsupported `method` is provided,
        if `fidelity` is not a discrete variable,
        or if `fidelity_distribution` is not provided.
    """

    if method != "latin_hypercube":
        raise ValueError(f"Unsupported sampling method: {method}")

    if (
        "fidelity" not in dimensions_dict
        or dimensions_dict["fidelity"][0] != "discrete"
    ):
        raise ValueError(
            "fidelity must be a discrete " "variable for distribution control."
        )

    if fidelity_distribution is None:
        raise ValueError("fidelity_distribution must be provided.")

    dimension_names = list(dimensions_dict.keys())
    other_dims = [d for d in dimension_names if d != "fidelity"]

    fidelity_values = list(fidelity_distribution.keys())
    fidelity_proportions = list(fidelity_distribution.values())
    fidelity_sample_counts = {
        d: int(round(num_samples * p))
        for d, p in zip(fidelity_values, fidelity_proportions)
    }

    total_assigned = sum(fidelity_sample_counts.values())
    while total_assigned != num_samples:
        max_key = max(fidelity_sample_counts, key=fidelity_sample_counts.get)
        fidelity_sample_counts[max_key] += num_samples - total_assigned
        total_assigned = sum(fidelity_sample_counts.values())

    final_samples = []

    for fidelity_value, count in fidelity_sample_counts.items():
        if count == 0:
            continue

        sampler = qmc.LatinHypercube(d=len(other_dims), seed=seed)
        samples = sampler.random(n=count)

        scaled_samples = np.zeros((count, len(dimension_names)))

        for i, dim_name in enumerate(other_dims):
            dim_type, dim_values = dimensions_dict[dim_name]
            dim_idx = dimension_names.index(dim_name)

            if dim_type == "continuous":
                low, high = dim_values
                scaled_samples[:, dim_idx] = qmc.scale(
                    samples[:, [i]], low, high
                ).flatten()
            elif dim_type == "discrete":
                allowed_values = np.array(dim_values)
                indices = (
                    np.floor(qmc.scale(samples[:, [i]], 0, len(allowed_values)))
                    .astype(int)
                    .flatten()
                )
                indices = np.clip(indices, 0, len(allowed_values) - 1)
                scaled_samples[:, dim_idx] = allowed_values[indices]
            else:
                raise ValueError(f"Invalid dimension type: {dim_type}")

        fidelity_index = dimension_names.index("fidelity")
        scaled_samples[:, fidelity_index] = fidelity_value

        if constraints:
            feeding_dims = constraints.get("feeding_dims", [])
            feeding_max = constraints.get("feeding_max", 0)
            if fidelity_value == 0:
                for row in scaled_samples:
                    if "feeding1" in dimension_names:
                        row[dimension_names.index("feeding1")] = 0
                    if "feeding3" in dimension_names:
                        row[dimension_names.index("feeding3")] = 0
                    if "feeding2" in dimension_names:
                        row[dimension_names.index("feeding2")] = feeding_max
            else:
                feeding_indices = [
                    dimension_names.index(dim)
                    for dim in feeding_dims
                    if dim in dimension_names
                ]
                if feeding_indices:
                    feeding_values = np.sum(
                        scaled_samples[:, feeding_indices], axis=1, keepdims=True
                    )
                    feeding_values[feeding_values == 0] = 1
                    scaled_samples[:, feeding_indices] *= feeding_max / feeding_values

        final_samples.append(scaled_samples)

    final_samples = np.vstack(final_samples)

    return final_samples[:num_samples]


def standardize_mixed_tensor(X, X_mean, X_std, catergorical_dim=-1):
    """
    Standardizes a mixed-type tensor containing both
    continuous and categorical features.

    Parameters
    ----------
    X : torch.Tensor
        The mixed-type input tensor.
    X_mean : np.ndarray or torch.Tensor
        The mean of the continuous features used for standardization.
    X_std : np.ndarray or torch.Tensor
        The standard deviation of the continuous features
        used for standardization.
    catergorical_dim : int, optional
        The dimension index of the categorical feature. Defaults to -1.

    Returns
    -------
    torch.Tensor
        The standardized mixed-type tensor.
    """
    X_categorical = X[:, catergorical_dim].unsqueeze(-1)
    X_continuous = X[:, :catergorical_dim]
    X_standardized = (X_continuous - X_mean) / X_std
    return torch.hstack((X_standardized, X_categorical))


def destandardize_mixed_tensor(X, X_mean, X_std, catergorical_dim=-1):
    """
    Destandardizes a mixed-type tensor containing both
    continuous and categorical features.

    Parameters
    ----------
    X : torch.Tensor
        The mixed-type input tensor to destandardize.
    X_mean : np.ndarray or torch.Tensor
        The mean of the continuous features used for destandardization.
    X_std : np.ndarray or torch.Tensor
        The standard deviation of the continuous features
        used for destandardization.
    catergorical_dim : int, optional
        The dimension index of the categorical feature. Defaults to -1.

    Returns
    -------
    torch.Tensor
        The destandardized mixed-type tensor.
    """
    X_categorical = X[:, catergorical_dim].unsqueeze(-1)
    X_continuous = X[:, :catergorical_dim]
    X_destandardized = X_continuous * X_std + X_mean
    return torch.hstack((X_destandardized, X_categorical))
