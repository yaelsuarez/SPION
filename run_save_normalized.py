#!/usr/bin/env python3
"""Save the Strategy-B normalised data per segmentation, and plot the feature space.

Uses the normalisation already computed in ``i0_rs/normalized/strategy_B``.
Nothing is refitted, no normalisation factor is recalculated, and no existing
result is modified.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_ROOT  # noqa: E402

import numpy as np  # noqa: E402

MRI = DATA_ROOT
FIT_ROOT = MRI / "i0_rs" / "Segmentations"
STRATEGY_B = MRI / "i0_rs" / "normalized" / "strategy_B"
OUT_DATA = MRI / "Normalized_data_i0_rs" / "Segmentations"
OUT_FIGS = MRI / "i0_rs" / "model_diagnostics" / "normalized_feature_space"

SEED = 42
MAX_PER_SEGMENT = 40_000
SAMPLE_ORDER = ("Syringes", "3D", "Pill1", "Pill2")

#: Volume arrays saved per segmentation, with their dtypes.
VOLUME_FIELDS = {
    "I0_raw": ("I0_selected", np.float32),
    "I0_normalized": ("I0_normalized", np.float32),
    "rs_selected": ("rs_selected", np.float32),
    "B_selected": ("B_selected", np.float32),
    "quality_ok": ("quality_ok", np.uint8),
    "flag_high_rs": ("flag_high_rs", np.uint8),
    "flag_low_information": ("flag_low_information", np.uint8),
    "model_selected": ("model_selected", np.uint8),
    "n_informative": ("n_informative", np.int16),
}


def main() -> int:
    import echoviewer  # noqa: F401
    import pandas as pd
    from echoviewer.diagnostics import _figure, _save, subsample

    started = time.time()
    saved: list[Path] = []
    written_bytes = 0
    per_sample: dict[str, dict[str, dict]] = {}

    # ---- 1. save per-segmentation normalised data ----------------------
    print("Saving normalised data")
    for sample in SAMPLE_ORDER:
        source = STRATEGY_B / sample / "voxels.csv.gz"
        if not source.is_file():
            print(f"  {sample}: no strategy_B output, skipped")
            continue
        table = pd.read_csv(source)
        factor = float(table["normalization_factor"].iloc[0])
        per_sample[sample] = {}

        for seg_name, block in table.groupby("segmentation", sort=True):
            fit_meta = json.loads((FIT_ROOT / sample / seg_name / "metadata.json").read_text())
            shape = tuple(fit_meta["shape"])
            folder = OUT_DATA / sample / seg_name
            folder.mkdir(parents=True, exist_ok=True)

            z = block["z"].to_numpy(np.intp)
            y = block["y"].to_numpy(np.intp)
            x = block["x"].to_numpy(np.intp)

            # Full-size volumes: values inside the segmentation, 0 outside.
            # Saved compressed - these arrays are ~99% zeros, so npz is about
            # twenty times smaller than npy for identical content.
            arrays = {}
            for name, (column, dtype) in VOLUME_FIELDS.items():
                volume = np.zeros(shape, dtype=dtype)
                volume[z, y, x] = block[column].to_numpy(dtype)
                arrays[name] = volume
            np.savez_compressed(folder / "volumes.npz", **arrays)
            del arrays

            keep = ["sample", "segmentation", "material", "x", "y", "z",
                    "rs_selected", "I0_selected", "I0_normalized", "B_selected",
                    "model_selected", "n_informative", "flag_high_rs",
                    "flag_low_information", "quality_ok", "normalization_factor"]
            block[keep].rename(columns={"I0_selected": "I0_raw"}).to_csv(
                folder / "voxels.csv.gz", index=False, float_format="%.6g",
                compression="gzip")

            material = str(block["material"].iloc[0])
            metadata = {
                "sample": sample, "segmentation": seg_name, "material": material,
                "shape": list(shape), "shape_order": ["z", "y", "x"],
                "voxel_spacing_zyx_mm": fit_meta["voxel_spacing"],
                "axis_map": {"x": "column (i)", "y": "row (j)", "z": "slice (k)"},
                "echo_times_ms": fit_meta["echo_times"],
                "n_voxels": int(len(block)),
                "n_quality_ok": int(block["quality_ok"].sum()),
                "n_flag_high_rs": int(block["flag_high_rs"].sum()),
                "n_flag_low_information": int(block["flag_low_information"].sum()),
                "normalization": {
                    "strategy": "B",
                    "factor": factor,
                    "definition": "I0_normalized = I0_raw / factor",
                    "method": ("sample scale from a two-way log-additive fit of median "
                               "I0 over (sample, concentration); for Pill1 the factor is "
                               "transferred, since it has no 0-SPION segmentation"),
                    "source": str(STRATEGY_B / sample / "metadata.json"),
                },
                "parameters": {
                    "representation": "AIC-selected",
                    "rule": ("Model A parameters with B=0 where AIC_A <= AIC_B, "
                             "Model B parameters otherwise"),
                    "model_selected": {"1": "Model A", "2": "Model B"},
                },
                "quality_flags": {
                    "flag_high_rs": "rs_selected > 0.15 1/ms",
                    "flag_low_information": ("fewer than 4 echoes above 2x the voxel "
                                             "noise floor; unreliable for slowly "
                                             "decaying materials such as water"),
                    "quality_ok": "neither flag set",
                },
                "files": {"volumes.npz": list(VOLUME_FIELDS),
                          "voxels.csv.gz": keep},
                "reconstruction": ("volume = np.zeros(shape, np.float32); "
                                   "volume[df.z, df.y, df.x] = df[<column>]"),
                "source_fits": str(FIT_ROOT / sample / seg_name),
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            (folder / "metadata.json").write_text(json.dumps(metadata, indent=2))

            size = sum(p.stat().st_size for p in folder.iterdir())
            written_bytes += size
            per_sample[sample][seg_name] = {
                "material": material,
                "rs": block["rs_selected"].to_numpy(np.float32),
                "I0n": block["I0_normalized"].to_numpy(np.float32),
                "n": len(block),
            }
            print(f"  {sample}/{seg_name:24s} {len(block):>7,} voxels  {size/1e6:6.1f} MB")
            del block
        del table

    # ---- 2. feature-space figures --------------------------------------
    print("\nPlotting")
    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    from matplotlib import colormaps

    # Shared limits so the four sample plots are directly comparable.
    every_rs = np.concatenate([d["rs"] for s in per_sample.values() for d in s.values()])
    every_i0 = np.concatenate([d["I0n"] for s in per_sample.values() for d in s.values()])
    limits = ((0, float(np.percentile(every_rs, 99.9)) * 1.05),
              (0, float(np.percentile(every_i0, 99.9)) * 1.05))
    del every_rs, every_i0

    palette = colormaps["tab10"]
    report = []

    def draw(axes, sample):
        rng = np.random.default_rng(SEED)
        total = drawn = 0
        for index, (seg_name, entry) in enumerate(sorted(per_sample[sample].items())):
            total += entry["n"]
            x, y = subsample([entry["rs"], entry["I0n"]], MAX_PER_SEGMENT, rng)
            drawn += x.size
            axes.scatter(x, y, s=3, alpha=0.18, linewidths=0,
                         color=palette(index % 10),
                         label=f"{seg_name} (n={entry['n']:,})")
        axes.set_xlabel("rs_selected (1/ms)")
        axes.set_ylabel("I0_normalized")
        axes.set_xlim(*limits[0])
        axes.set_ylim(*limits[1])
        axes.axhline(1.0, color="black", ls="--", lw=0.8)
        axes.grid(alpha=0.25)
        return total, drawn

    for sample in per_sample:
        figure = _figure(figsize=(9.5, 8))
        axes = figure.subplots()
        total, drawn = draw(axes, sample)
        report.append((sample, total, drawn))
        axes.set_title(f"{sample} — model-selected rs vs Strategy-B normalised I0\n"
                       f"{drawn:,} of {total:,} voxels plotted, seed {SEED}", fontsize=11)
        legend = axes.legend(markerscale=5, fontsize=8)
        for handle in legend.legend_handles:
            handle.set_alpha(1.0)
        _save(figure, OUT_FIGS / f"rs_vs_I0_normalized_{sample}.png", saved)
        print(f"  {sample}: {drawn:,} of {total:,} plotted")

    # Combined: one colour per segmentation across all samples, marker per sample
    figure = _figure(figsize=(13, 10))
    axes = figure.subplots()
    markers = {"Syringes": "o", "3D": "^", "Pill1": "s", "Pill2": "D"}
    rng = np.random.default_rng(SEED)
    total = drawn = 0
    colour_index = 0
    for sample in per_sample:
        for seg_name, entry in sorted(per_sample[sample].items()):
            total += entry["n"]
            x, y = subsample([entry["rs"], entry["I0n"]], MAX_PER_SEGMENT, rng)
            drawn += x.size
            axes.scatter(x, y, s=3, alpha=0.18, linewidths=0,
                         color=palette(colour_index % 10),
                         marker=markers.get(sample, "o"),
                         label=f"{sample} / {seg_name} (n={entry['n']:,})")
            colour_index += 1
    axes.set_xlabel("rs_selected (1/ms)")
    axes.set_ylabel("I0_normalized")
    axes.set_xlim(*limits[0])
    axes.set_ylim(*limits[1])
    axes.axhline(1.0, color="black", ls="--", lw=0.8)
    axes.grid(alpha=0.25)
    axes.set_title("All samples — model-selected rs vs Strategy-B normalised I0\n"
                   f"colour = segmentation, marker = sample; {drawn:,} of {total:,} "
                   f"voxels plotted, seed {SEED}", fontsize=12)
    legend = axes.legend(markerscale=5, fontsize=7, ncol=2, loc="upper right")
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
    _save(figure, OUT_FIGS / "rs_vs_I0_normalized_all_samples.png", saved)
    report.append(("all samples", total, drawn))
    print(f"  combined: {drawn:,} of {total:,} plotted")

    print("\n" + "=" * 70)
    print(f"Normalised data : {OUT_DATA}")
    print(f"                  {written_bytes/1e6:.0f} MB in "
          f"{sum(len(v) for v in per_sample.values())} segmentations")
    print(f"Figures         : {OUT_FIGS}  ({len(saved)} PNGs)")
    print(f"\n{'plot':16s} {'total voxels':>14s} {'plotted':>10s} {'fraction':>9s}  seed")
    for name, total, drawn in report:
        print(f"{name:16s} {total:>14,} {drawn:>10,} {100*drawn/total:>8.1f}%  {SEED}")
    print(f"\nelapsed {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
