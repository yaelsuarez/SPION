#!/usr/bin/env python3
"""Assemble the complete Pill2 volume for later classification.

The voxel-wise fit for this volume already exists under ``i0_rs/Volumes/Pill2``,
produced by the same code path and the same rules as the segmented data:
Model A ``I0*exp(-rs*TE)``, Model B ``I0*exp(-rs*TE)+B``, model chosen by lower
AIC with ties to Model A, valid voxels = first-echo intensity above the
volume's Otsu level. Refitting would repeat that work and could only reproduce
the same numbers, so this script derives the requested maps from the stored
result instead.

What it adds beyond the existing output:

* ``AIC_A`` and ``AIC_B`` as full-volume maps (they existed only in the CSV);
* the model-selected representation ``I0_selected``/``rs_selected``/``B_selected``;
* the informative-echo count and quality flags, computed from the DICOM data,
  matching the definition used for the segmentations.

Nothing under ``i0_rs/Volumes`` is modified. No normalisation, no labels used.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MRI = DATA_ROOT
SOURCE = MRI / "i0_rs" / "Volumes" / "Pill2"
VOLUMES = MRI / "Volumes"
OUT = MRI / "i0_rs" / "Volume_for_testing" / "Pill2"

SAMPLE = "Pill2"
HIGH_RS = 0.15          # same cut as the segmentation quality flags
MIN_INFORMATIVE = 4     # Model B has three parameters
NOISE_FACTOR = 2.0

#: Output map name -> (source column in the stored CSV, dtype)
MAPS = {
    "I0_A": ("I0_noB", np.float32),
    "rs_A": ("rs_noB", np.float32),
    "AIC_A": ("AIC_noB", np.float32),
    "R2_A": ("R2_noB", np.float32),
    "RMSE_A": ("RMSE_noB", np.float32),
    "I0_B": ("I0_B", np.float32),
    "rs_B": ("rs_B", np.float32),
    "B_B": ("B", np.float32),
    "AIC_B": ("AIC_B", np.float32),
    "R2_B": ("R2_B", np.float32),
    "RMSE_B": ("RMSE_B", np.float32),
    "model_selected": ("model_selected", np.int8),
    "fit_success": ("fit_success", np.uint8),
}


def main() -> int:
    import echoviewer  # noqa: F401
    from echoviewer.series import EchoSeries

    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    source_meta = json.loads((SOURCE / "metadata.json").read_text())
    shape = tuple(source_meta["shape"])
    echo_times = source_meta["echo_times"]
    print(f"Source fit: {SOURCE}")
    print(f"  shape {shape}, {len(echo_times)} echo times, "
          f"{source_meta['n_voxels_evaluated']:,} evaluated voxels")
    print(f"  valid-voxel rule: {source_meta['valid_voxel_rule']}")
    print(f"  model rule: {source_meta['model_choice_criterion']}\n")

    frame = pd.read_csv(SOURCE / "voxel_fit_data.csv")
    frame = frame[frame["fit_success"] > 0].reset_index(drop=True)
    z = frame["z"].to_numpy(np.intp)
    y = frame["y"].to_numpy(np.intp)
    x = frame["x"].to_numpy(np.intp)
    print(f"Loaded {len(frame):,} successfully fitted voxels")

    # The stored model_selected was computed at full precision during fitting;
    # the CSV rounds AIC to six significant figures, which flips near-ties. Use
    # the stored decision and report how far the two disagree.
    recomputed = np.where(frame["AIC_noB"].to_numpy() <= frame["AIC_B"].to_numpy(), 1, 2)
    disagreements = int((recomputed != frame["model_selected"].to_numpy()).sum())
    chose_a = frame["model_selected"].to_numpy() == 1
    print(f"  AIC rule recomputed from the rounded CSV differs on {disagreements} "
          f"voxels ({100*disagreements/len(frame):.4f}%); stored decision used")

    # ---- informative echoes from the DICOM data -------------------------
    print("\nCounting informative echoes from the DICOM volumes")
    series = EchoSeries.from_sample(VOLUMES, SAMPLE, cache_size=2)
    try:
        curves = np.zeros((len(frame), len(series)), dtype=np.float32)
        for column in range(len(series)):
            curves[:, column] = series.volume(column).data[z, y, x]
    finally:
        series.close()
    floor = np.median(curves[:, -5:], axis=1, keepdims=True)
    n_informative = (curves > NOISE_FACTOR * np.maximum(floor, 1e-6)).sum(axis=1)
    del curves

    flag_high_rs = np.where(chose_a, frame["rs_noB"], frame["rs_B"]) > HIGH_RS
    flag_low_information = n_informative < MIN_INFORMATIVE
    quality_ok = ~(flag_high_rs | flag_low_information)

    # ---- build and save the maps ----------------------------------------
    print("\nWriting maps")
    written = []

    def save(name, values, dtype):
        volume = np.zeros(shape, dtype=dtype)
        volume[z, y, x] = values.astype(dtype)
        np.save(OUT / f"{name}.npy", volume)
        written.append(f"{name}.npy")
        del volume

    for name, (column, dtype) in MAPS.items():
        save(name, frame[column].to_numpy(), dtype)

    # model-selected representation: Model A's own parameters where A wins
    i0_selected = np.where(chose_a, frame["I0_noB"], frame["I0_B"])
    rs_selected = np.where(chose_a, frame["rs_noB"], frame["rs_B"])
    b_selected = np.where(chose_a, 0.0, frame["B"])
    save("I0_selected", i0_selected, np.float32)
    save("rs_selected", rs_selected, np.float32)
    save("B_selected", b_selected, np.float32)

    save("n_informative", n_informative, np.int16)
    save("flag_high_rs", flag_high_rs, np.uint8)
    save("flag_low_information", flag_low_information, np.uint8)
    save("quality_ok", quality_ok, np.uint8)

    # ---- per-voxel table -------------------------------------------------
    table = pd.DataFrame({
        "x": x.astype(np.int32), "y": y.astype(np.int32), "z": z.astype(np.int32),
        "I0_A": frame["I0_noB"], "rs_A": frame["rs_noB"], "AIC_A": frame["AIC_noB"],
        "R2_A": frame["R2_noB"], "RMSE_A": frame["RMSE_noB"],
        "I0_B": frame["I0_B"], "rs_B": frame["rs_B"], "B_B": frame["B"],
        "AIC_B": frame["AIC_B"], "R2_B": frame["R2_B"], "RMSE_B": frame["RMSE_B"],
        "model_selected": frame["model_selected"].astype(np.int8),
        "I0_selected": i0_selected, "rs_selected": rs_selected, "B_selected": b_selected,
        "n_informative": n_informative.astype(np.int16),
        "flag_high_rs": flag_high_rs.astype(np.int8),
        "flag_low_information": flag_low_information.astype(np.int8),
        "quality_ok": quality_ok.astype(np.int8),
        "fit_success": frame["fit_success"].astype(np.int8),
    })
    table.to_csv(OUT / "voxel_data.csv.gz", index=False, float_format="%.6g",
                 compression="gzip")
    written.append("voxel_data.csv.gz")

    # ---- metadata ---------------------------------------------------------
    metadata = {
        "sample": SAMPLE, "source": "complete volume (all valid voxels, not segmented)",
        "shape": list(shape), "shape_order": ["z", "y", "x"],
        "axis_map": {"x": "column (i)", "y": "row (j)", "z": "slice (k)"},
        "voxel_spacing_zyx_mm": source_meta["voxel_spacing"],
        "echo_times_ms": echo_times,
        "n_voxels": int(len(frame)),
        "valid_voxel_rule": source_meta["valid_voxel_rule"],
        "models": source_meta["models"],
        "model_selection": {
            "rule": "Model A if AIC_A <= AIC_B, else Model B",
            "encoding": {"1": "Model A", "2": "Model B"},
            "selected_parameters": ("Model A's own I0/rs with B_selected = 0 when A wins; "
                                    "Model B's I0/rs/B when B wins"),
            "note": ("the stored full-precision decision from the original fit is used; "
                     f"recomputing from the rounded CSV would differ on {disagreements} voxels"),
        },
        "quality_flags": {
            "flag_high_rs": f"rs_selected > {HIGH_RS} 1/ms",
            "flag_low_information": (f"fewer than {MIN_INFORMATIVE} echoes above "
                                     f"{NOISE_FACTOR}x the voxel noise floor (median of the "
                                     "last five echoes); unreliable for slowly decaying "
                                     "material such as water"),
            "quality_ok": "neither flag set",
        },
        "provenance": {
            "fit_results_reused_from": str(SOURCE),
            "refitted": False,
            "reason": ("the complete volume was already fitted with the identical "
                       "procedure; refitting could only reproduce the same numbers"),
            "dicom_source": str(VOLUMES / SAMPLE),
        },
        "not_done_yet": ["normalisation", "classification", "any use of Pill2 labels"],
        "files": written,
        "reconstruction": "volume = np.load('<map>.npy')  # already full size, 0 outside",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # ---- report -----------------------------------------------------------
    n = len(frame)
    n_a, n_b = int(chose_a.sum()), int((~chose_a).sum())
    print("\n" + "=" * 70)
    print(f"Pill2 complete volume — {n:,} valid voxels of {int(np.prod(shape)):,} "
          f"({100*n/np.prod(shape):.1f}% of the volume)")
    print("=" * 70)
    print(f"\nModel A selected: {n_a:>9,}  ({100*n_a/n:6.2f}%)")
    print(f"Model B selected: {n_b:>9,}  ({100*n_b/n:6.2f}%)")

    print("\nParameter statistics (all valid voxels)")
    print(f"  {'parameter':14s}{'median':>12s}{'q1':>12s}{'q3':>12s}{'min':>12s}{'max':>12s}")
    for name, values in (("I0_A", frame["I0_noB"]), ("rs_A", frame["rs_noB"]),
                         ("I0_B", frame["I0_B"]), ("rs_B", frame["rs_B"]),
                         ("B_B", frame["B"]), ("I0_selected", i0_selected),
                         ("rs_selected", rs_selected), ("B_selected", b_selected)):
        v = np.asarray(values, dtype=np.float64)
        print(f"  {name:14s}{np.median(v):12.4f}{np.percentile(v,25):12.4f}"
              f"{np.percentile(v,75):12.4f}{v.min():12.4f}{v.max():12.4f}")

    print("\nFit quality")
    for name, values in (("R2_A", frame["R2_noB"]), ("R2_B", frame["R2_B"]),
                         ("RMSE_A", frame["RMSE_noB"]), ("RMSE_B", frame["RMSE_B"])):
        v = np.asarray(values, dtype=np.float64)
        print(f"  {name:8s} median {np.median(v):9.4f}   "
              f"IQR [{np.percentile(v,25):9.4f}, {np.percentile(v,75):9.4f}]")

    print("\nQuality flags")
    print(f"  high rs (> {HIGH_RS})          {int(flag_high_rs.sum()):>9,}  "
          f"({100*flag_high_rs.mean():5.2f}%)")
    print(f"  low information (< {MIN_INFORMATIVE} echoes) {int(flag_low_information.sum()):>9,}  "
          f"({100*flag_low_information.mean():5.2f}%)")
    print(f"  pass both                  {int(quality_ok.sum()):>9,}  "
          f"({100*quality_ok.mean():5.2f}%)")
    print(f"  informative echoes: median {int(np.median(n_informative))}, "
          f"p10 {int(np.percentile(n_informative,10))}, "
          f"p90 {int(np.percentile(n_informative,90))}")

    size = sum(p.stat().st_size for p in OUT.iterdir())
    print(f"\nWritten to {OUT}")
    print(f"  {len(written)} files, {size/1e6:.0f} MB")
    print(f"elapsed {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
