#!/usr/bin/env python3
"""Rewrite the two rs-vs-I0 workbooks keeping a 10% subsample, for Excel's sake.

The sheets are rebuilt from the same stored sources the full workbooks were
transcribed from, then thinned. Sampling is stratified by segmentation with
seed 42, so every segmentation - and therefore every class - keeps a tenth of
its voxels; a flat random draw would thin the small classes unevenly.

Values themselves are never altered, and failed_parameters.xlsx is not touched.
The full-resolution workbooks are regenerable at any time by rerunning
results_workbooks.py against the untouched source stores.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from data_treatment.results_workbooks import (OUT, raw_and_normalized,  # noqa: E402
                                              write_workbook)

FRACTION = 0.10
SEED = 42


def thin(sheets):
    out = {}
    for name, frame in sheets.items():
        kept = (frame.groupby("segmentation", group_keys=False)
                     .apply(lambda g: g.sample(frac=FRACTION, random_state=SEED)))
        kept = kept.sort_index().reset_index(drop=True)
        out[name] = kept
        print(f"  {name:9s} {len(frame):>8,} -> {len(kept):>7,} rows "
              f"({100*len(kept)/len(frame):.1f}%), {kept.material.nunique()} classes kept")
    return out


def main() -> int:
    started = time.time()
    raw_sheets, norm_sheets = raw_and_normalized()

    note = [("Subsample", f"{FRACTION:.0%} of voxels, stratified by segmentation, seed {SEED}"),
            ("Why", "the full workbook was too large to open comfortably in Excel"),
            ("Values", "unchanged; rows were removed, never edited"),
            ("Full data", "regenerate with data_treatment/results_workbooks.py; the source "
                          "stores were never modified")]

    print("\nrs_vs_i0.xlsx")
    write_workbook(OUT / "rs_vs_i0.xlsx", thin(raw_sheets), [
        ("Content", "Raw rs vs I0 per voxel, one sheet per sample, all classes"),
        ("Source", "MRI/Normalized_data_i0_rs/Segmentations/<sample>/<segmentation>/voxels.csv.gz"),
        ("I0 column", "I0_raw - the fitted amplitude, independent of any normalisation"),
        ("rs column", "rs_selected - the AIC-selected decay rate, 1/ms"),
        ("material", "taken from the segmentation folder name, not the CSV column"),
        *note,
        ("Created", time.strftime("%Y-%m-%d %H:%M:%S")),
    ])
    size = (OUT / "rs_vs_i0.xlsx").stat().st_size / 1e6
    print(f"  wrote rs_vs_i0.xlsx ({size:.0f} MB)")

    factors = {s: float(f.normalization_factor.iloc[0]) for s, f in norm_sheets.items()}
    print("\nrs_vs_i0_normalized.xlsx")
    write_workbook(OUT / "rs_vs_i0_normalized.xlsx", thin(norm_sheets), [
        ("Content", "rs vs I0_normalized per voxel, one sheet per sample, all classes"),
        ("Normalisation", "final label-light scheme"),
        ("Factors", ", ".join(f"{s} {v:g}" for s, v in factors.items())),
        ("Source, training", "MRI/Normalized_data_i0_rs/Segmentations_training/"),
        ("Source, Pill2", "MRI/i0_rs/Volume_for_testing/Pill2_segmentation_complete/"),
        ("Not used", "Normalized_data_i0_rs/Segmentations I0_normalized - superseded "
                     "Strategy-B factors 6452.85 / 1229.50 / 1304.21 / 2011.36"),
        ("I0_residual", "I0_normalized minus f(rs), the degree-2 polynomial fitted on "
                        "Syringes + 3D + Pill1 only"),
        *note,
        ("Created", time.strftime("%Y-%m-%d %H:%M:%S")),
    ])
    size = (OUT / "rs_vs_i0_normalized.xlsx").stat().st_size / 1e6
    print(f"  wrote rs_vs_i0_normalized.xlsx ({size:.0f} MB)")
    print(f"\nfailed_parameters.xlsx untouched")
    print(f"elapsed {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
