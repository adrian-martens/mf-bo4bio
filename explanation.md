# mfbo4bio -- Module Explanation

This document describes the purpose and responsibility of each module in the `mfbo4bio` package, which implements multi-fidelity Bayesian optimization for bioprocess (CHO cell culture) optimization using a virtual laboratory.

## High-Level Architecture

```
run/                        CLI driver scripts for batch experiments
src/mfbo4bio/
    __init__.py             Public API
    config.py               Typed configuration dataclasses
    presets.py              Frozen preset configurations for batch runs
    pipeline.py             Top-level runners (BO and industrial baselines)
    bo_core/                Core BO loop
        engine.py           Main BO iteration loop
        acquisition.py      Acquisition function construction
        fidelity_selection.py   Cross-fidelity scoring
        candidate_generation.py Candidate proposal and optimization
        state.py            Mutable run state tracking
    utils.py                GP models, data scaling, sampling
    optimization.py         Custom acquisition optimizer (fidelity-0 batches)
    virtual_lab.py          CHO bioreactor simulation (the objective function)
    conditions_data.py      Clone kinetic parameters and domain constants
    industrial_methods.py   Classical DOE baseline methods
    clone_creator.py        Synthetic clone parameter generator
    randomness.py           Deterministic seed management
    validation.py           Input data guards
    results.py              Output path management and JSON serialization
    mfbo_GIBBON.py          GIBBON entry point
    mfbo_qLogEI.py          qLogEI entry point
    mfbo_qUCB.py            qUCB entry point
tests/                      Pytest test suite
notebooks/                  Analysis and visualization notebooks
```

---

## Package Root

### `__init__.py`

Exposes the two public entry points: `run_bo` (Bayesian optimization) and `run_industrial` (classical DOE baselines).

---

## Configuration

### `config.py`

Defines three nested dataclasses that fully specify a run:

- **`ExperimentConfig`** -- physical domain settings: clone distribution (`alpha` or `beta`), MBR fidelity level (1--9), feeding maximum, temperature and pH bounds.
- **`BOConfig`** -- optimization settings: number of BO iterations, task representation (`HYBRID` entity-embedding GP or `ICM_WRAPPED` multi-task GP), embedding dimension, UCB beta, per-fidelity batch sizes and costs.
- **`RunConfig`** -- top-level run descriptor: BO method name (`qUCB`, `qLogEI`, `GIBBON`, or `industrial`), output name, random seed, results directory, logging level, reproducibility flag.

### `presets.py`

Frozen dataclasses (`BOScenarioPreset`, `IndustrialPreset`) holding default scenario grids for batch experiment sweeps. Used by the `run/` driver scripts to enumerate all (method, seed, distribution, ...) combinations.

---

## Pipeline and Execution

### `pipeline.py`

Contains the two high-level runners:

- **`run_bo(config)`** -- validates the method and delegates to `run_bo_engine` in `bo_core/engine.py`.
- **`run_industrial(config)`** -- executes a classical three-stage industrial DOE workflow: (1) low-fidelity screening across all clones, (2) mid-fidelity optimization on the best clone, (3) high-fidelity confirmation runs around the best conditions. Saves results as JSON.

### `mfbo_GIBBON.py` / `mfbo_qLogEI.py` / `mfbo_qUCB.py`

Thin CLI entry points. Each builds a `RunConfig` with the respective `method` and calls `run_bo`. They also expose an `argparse` interface for command-line invocation.

---

## BO Core (`bo_core/`)

### `engine.py`

The main multi-fidelity BO loop (`run_bo_engine`). Each iteration:

1. Builds standardized bounds for the search space.
2. For GIBBON: generates a 512-point Latin hypercube candidate set at the highest fidelity and defines a `project_to_fidelity` function.
3. Constructs the acquisition function via `build_acquisition`.
4. Proposes candidate batches at each fidelity level (`propose_batch_for_fidelity`).
5. Scores each fidelity's batch (`gibbon_score` or `correlation_weighted_score`).
6. Selects the best fidelity, evaluates the batch in the virtual lab, updates the GP model.

Also handles initial Latin hypercube sampling, GP training, state tracking, cost budgeting (40 000 EUR cap), and JSON output.

### `acquisition.py`

Factory function `build_acquisition` that wires up the appropriate BoTorch acquisition function depending on the method:

- **qUCB** -- `qUpperConfidenceBound` with a Sobol MC sampler.
- **qLogEI** -- `qLogExpectedImprovement` with a Sobol MC sampler.
- **GIBBON** -- `qMultiFidelityLowerBoundMaxValueEntropy` with Gumbel max-value sampling and a no-op cost utility (cost normalization is handled externally in `gibbon_score`).

### `fidelity_selection.py`

Functions that score candidate batches across fidelities to decide which fidelity to evaluate:

- **`correlation_weighted_score`** (qUCB, qLogEI) -- weights acquisition values by the posterior correlation between the candidate fidelity and the highest fidelity, then normalizes by batch size and cost.
- **`gibbon_score`** (GIBBON) -- normalizes the raw GIBBON acquisition values by batch size and cost for fair cross-fidelity comparison.

### `candidate_generation.py`

Handles candidate proposal for a single fidelity level:

- **`build_bounds`** -- constructs standardized optimization bounds (7D: temperature, pH, three feedings, fidelity, clone).
- **`build_inequality_constraints`** -- enforces the total feeding sum constraint in min-max scaled space.
- **`propose_batch_for_fidelity`** -- for fidelity 0, uses `custom_optimization` (grid-based sampling); for higher fidelities, uses BoTorch's `optimize_acqf` with sequential greedy optimization, inequality constraints, and a sampling fallback.
- Post-processing for GIBBON: rounds the clone dimension and re-evaluates per-point acquisition values.

### `state.py`

- **`BOState`** -- mutable dataclass tracking the full history of a BO run: all inputs/outputs, scaling statistics, per-iteration best values, batches, selected fidelities, and cumulative cost.
- **`BORunResult`** -- immutable result container returned by `run_bo_engine`.

---

## GP Models, Scaling, and Sampling (`utils.py`)

This is the largest utility module and provides:

- **`CustomMultiTaskGP`** (HYBRID model) -- a `SingleTaskGP` with a product kernel: `ScaleKernel(RBFKernel)` over the continuous dimensions multiplied by an `EntityEmbeddingKernel` over the categorical clone dimension. The embedding maps each clone to a learned vector in a configurable-dimension latent space.
- **`EntityEmbeddingKernel`** -- a GPyTorch kernel that computes covariance via Euclidean distance between learned clone embeddings.
- **`TaskFilteredMultiTaskGP`** (ICM_WRAPPED model) -- wraps BoTorch's `MultiTaskGP` (Intrinsic Coregionalization Model) and overrides `num_outputs` to 1 for compatibility with single-output acquisition functions.
- **`train_gp_model`** -- fits the GP using `fit_gpytorch_mll` with retry logic for optimization warnings.
- **Data scaling** -- `standardize_data`, `minmax_scale_data`, `standardize_target`, `unstandardize_y`, `standardize_mixed_tensor`, `destandardize_mixed_tensor`.
- **`sampling`** -- generates quasi-random samples (Latin hypercube, Sobol, factorial) in a mixed continuous/discrete space with fidelity distribution control and feeding constraints.

---

## Custom Acquisition Optimization (`optimization.py`)

`custom_optimization` handles acquisition optimization specifically for fidelity-0 (MTP) batches, where gradient-based optimization is impractical due to the constrained, low-dimensional structure. It generates candidates on a temperature/pH grid, assigns clones via Latin hypercube sampling, and evaluates the acquisition function sequentially with `set_X_pending` to account for batch diversity.

---

## Virtual Laboratory (`virtual_lab.py`)

The simulated objective function. The `EXPERIMENT` class implements a CHO (Chinese Hamster Ovary) cell culture kinetic model:

- ODE system with 8 state variables: product (P), total cells (X_T), viable cells (X_V), dead cells (X_D), glucose (G), glutamine (Q), lactate (L), ammonia (A).
- Temperature and pH effects on growth rate via Gaussian response functions.
- Reactor scale effects: MTP (scale_factor=0.8, growth_inhibition=0.7), MBR (1.0, 1.0), PILOT (1.4, 0.6).
- Segment-wise ODE integration with feeding events using `scipy.integrate.solve_ivp`.
- Noisy measurements via `measurement()` with reactor-dependent noise levels.

`conduct_experiment(X)` dispatches on input dimensionality (2D, 5D, 6D, or 7D) to set temperature, pH, feeding schedule, fidelity/reactor, and clone, then runs the simulation and returns product titers.

---

## Domain Data (`conditions_data.py`)

Contains the biological domain knowledge:

- `feeding_max = 50` -- maximum total feeding amount.
- `noise_level` -- per-reactor measurement noise (MTP, MBR, PILOT).
- `process_parameters_alpha` / `process_parameters_beta` -- dictionaries mapping 30 cell clones (`celltype_1` through `celltype_30`) to their kinetic and stoichiometric parameters (growth rates, yields, Monod constants, etc.).
- `get_cell_params(clone_distribution)` -- returns a pandas DataFrame of all clone parameters for a given distribution.

---

## Industrial Baselines (`industrial_methods.py`)

Implements classical Design of Experiments (DOE) sampling strategies used as baselines:

- **`factorial`** -- full factorial design with fidelity distribution control.
- **`quasi_mc_methods`** -- Latin hypercube and Sobol sequence sampling.
- **`sampling`** -- unified interface dispatching to the above methods.
- **`apply_feeding_constraints`** -- enforces feeding constraints (zero feeding at fidelity 0 except feeding2, total feeding normalization at higher fidelities).
- **`run`** -- convenience function that builds a `RunConfig` and calls `run_industrial`.

---

## Supporting Modules

### `randomness.py`

`SeedManager` derives deterministic child seeds from a base seed using SHA-256 hashing with named namespaces. Provides separate RNGs for NumPy, PyTorch, and Sobol sequences to ensure reproducibility without seed collision.

### `validation.py`

Input guard functions used throughout the BO loop:

- `ensure_non_empty_2d` -- checks array shape.
- `ensure_finite` -- checks for NaN/Inf values.
- `ensure_valid_clone_ids` -- checks clone indices are in range.
- `ensure_allowed_fidelities` -- checks fidelity values against allowed set.

### `results.py`

Output path construction and serialization:

- `project_root()` -- resolves the repository root.
- `bo_output_paths(config)` -- constructs output JSON paths based on method, clone distribution, and task representation.
- `industrial_output_path(config)` -- constructs output paths for industrial baselines.
- `save_json(payload, path)` -- serializes results (converting tensors/arrays to lists) and writes JSON.

### `clone_creator.py`

`generate_process_parameters` creates randomized kinetic parameters for synthetic cell clones and writes them to a Python file. Used to generate the `process_parameters_alpha` and `process_parameters_beta` dictionaries in `conditions_data.py`.

---

## Driver Scripts (`run/`)

### `run_single_bo_scenario.py`

CLI entry point for a single BO run. Parses command-line arguments (method, iterations, seed, MBR level, clone distribution, date, task representation) and dispatches to the appropriate `mfbo_*.run()` function.

### `run_all_bo_scenarios.py`

Batch runner that reads `DEFAULT_BO_PRESET`, enumerates all combinations of (method, MBR level, distribution, seed, date, task representation), and runs them in parallel using `multiprocessing.Pool`.

### `run_industrial_scenarios.py`

Runs industrial baselines (factorial, Latin hypercube, Sobol) across multiple seeds using `DEFAULT_INDUSTRIAL_PRESET`.
