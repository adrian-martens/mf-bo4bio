from mfbo4bio import industrial_methods

# This script runs multiple industrial approach experiments
# for a selected clone distribution (e.g. alpha)

clone_dist = "alpha"
platform_cond = True
date = "20250805"

for i in range(10):
    industrial_methods.run(
        output_name=f"factorial_{date}_{i+1}",
        n_iterations=15,
        sampling_method="factorial",
        seed=i,
        clone_distribution=clone_dist,
        platform_cond=platform_cond,
    )
    industrial_methods.run(
        output_name=f"lhs_{date}_{i+1}",
        n_iterations=15,
        sampling_method="latin_hypercube",
        seed=i,
        clone_distribution=clone_dist,
        platform_cond=platform_cond,
    )
    industrial_methods.run(
        output_name=f"sobol_{date}_{i+1}",
        n_iterations=15,
        sampling_method="sobol",
        seed=i,
        clone_distribution=clone_dist,
        platform_cond=platform_cond,
    )
