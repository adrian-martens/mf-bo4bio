from mfbo4bio import industrial_methods
from mfbo4bio.presets import DEFAULT_INDUSTRIAL_PRESET_ALPHA, DEFAULT_INDUSTRIAL_PRESET_BETA

# ── Settings ──────────────────────────────────────────────────────────────────
MTP_FEED_MODE = "none"   # "none" | "fixed_max" | "variable"
# ─────────────────────────────────────────────────────────────────────────────


def _run_preset(preset):
    for i in range(preset.repeats):
        industrial_methods.run(
            output_name=f"factorial_{preset.date}_{i+1}",
            sampling_method="factorial",
            seed=i,
            clone_distribution=preset.clone_dist,
            platform_cond=preset.platform_cond,
            mtp_feed_mode=MTP_FEED_MODE,
        )
        industrial_methods.run(
            output_name=f"lhs_{preset.date}_{i+1}",
            sampling_method="latin_hypercube",
            seed=i,
            clone_distribution=preset.clone_dist,
            platform_cond=preset.platform_cond,
            mtp_feed_mode=MTP_FEED_MODE,
        )
        industrial_methods.run(
            output_name=f"sobol_{preset.date}_{i+1}",
            sampling_method="sobol",
            seed=i,
            clone_distribution=preset.clone_dist,
            platform_cond=preset.platform_cond,
            mtp_feed_mode=MTP_FEED_MODE,
        )


def main() -> None:
    _run_preset(DEFAULT_INDUSTRIAL_PRESET_ALPHA)
    _run_preset(DEFAULT_INDUSTRIAL_PRESET_BETA)


if __name__ == "__main__":
    main()
