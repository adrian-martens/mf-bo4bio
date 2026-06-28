import argparse

from mfbo4bio import industrial_methods
from mfbo4bio.presets import DEFAULT_INDUSTRIAL_PRESET


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mtp_feed_mode",
        type=str,
        default="none",
        choices=["none", "fixed_max", "variable"],
    )
    args = parser.parse_args()

    preset = DEFAULT_INDUSTRIAL_PRESET
    for i in range(preset.repeats):
        industrial_methods.run(
            output_name=f"factorial_{preset.date}_{i+1}",
            n_iterations=21,
            sampling_method="factorial",
            seed=i,
            clone_distribution=preset.clone_dist,
            platform_cond=preset.platform_cond,
            mtp_feed_mode=args.mtp_feed_mode,
        )
        industrial_methods.run(
            output_name=f"lhs_{preset.date}_{i+1}",
            n_iterations=21,
            sampling_method="latin_hypercube",
            seed=i,
            clone_distribution=preset.clone_dist,
            platform_cond=preset.platform_cond,
            mtp_feed_mode=args.mtp_feed_mode,
        )
        industrial_methods.run(
            output_name=f"sobol_{preset.date}_{i+1}",
            n_iterations=21,
            sampling_method="sobol",
            seed=i,
            clone_distribution=preset.clone_dist,
            platform_cond=preset.platform_cond,
            mtp_feed_mode=args.mtp_feed_mode,
        )


if __name__ == "__main__":
    main()
