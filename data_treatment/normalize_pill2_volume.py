#!/usr/bin/env python3
"""Normalise the complete Pill2 volume with the established pipeline.

    I0_selected  ->  / factor  ->  I0_normalized  ->  - f(rs)  ->  I0_residual

Nothing is refitted: the full-volume Model A/B fits and the selected
representation already exist under ``Volume_for_testing/Pill2``.

Two choices are recorded explicitly because they matter for a held-out test set:

* **The scaling factor.** Strategy B's *fitted* scale for Pill2 (2011.36) came
  from a two-way fit that included Pill2's labelled concentrations. Pill2's
  *0-SPION reference* factor (2042.62) uses only the identity of the 0-SPION
  region, no concentration labels. The reference factor is used as primary, and
  the fitted scale is saved alongside so the 1.5% difference can be inspected.
* **f(rs).** Recomputed from Syringes + 3D + Pill1 only, exactly as before, and
  checked against the published coefficients. Pill2 never enters the fit. The
  coefficients are persisted this time so nothing downstream has to re-derive
  them.
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
SOURCE = MRI / "i0_rs" / "Volume_for_testing" / "Pill2"
NORMALIZED_SEG = MRI / "Normalized_data_i0_rs" / "Segmentations"
STRATEGY_A = MRI / "i0_rs" / "normalized" / "strategy_A" / "Pill2" / "metadata.json"
STRATEGY_B = MRI / "i0_rs" / "normalized" / "strategy_B" / "Pill2" / "metadata.json"
OUT = MRI / "i0_rs" / "Volume_for_testing" / "Pill2_normalized"

TRAINING = ("Syringes", "3D", "Pill1")     # Pill2 is the held-out test set
PUBLISHED_F = (-166.1185, 19.0491, 0.8232)


def main() -> int:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- inputs, none refitted ------------------------------------------
    meta = json.loads((SOURCE / "metadata.json").read_text())
    shape = tuple(meta["shape"])
    print(f"Source: {SOURCE}")
    print(f"  {meta['n_voxels']:,} valid voxels, shape {shape}, refitted: {meta['provenance']['refitted']}")

    rs = np.load(SOURCE / "rs_selected.npy")
    i0 = np.load(SOURCE / "I0_selected.npy")
    b_sel = np.load(SOURCE / "B_selected.npy")
    valid = np.load(SOURCE / "fit_success.npy") > 0
    quality_ok = np.load(SOURCE / "quality_ok.npy") > 0
    flag_high_rs = np.load(SOURCE / "flag_high_rs.npy") > 0
    flag_low_info = np.load(SOURCE / "flag_low_information.npy") > 0
    n_informative = np.load(SOURCE / "n_informative.npy")

    # ---- scaling factors, taken not recomputed ---------------------------
    reference_factor = float(json.loads(STRATEGY_A.read_text())["normalization_factor"])
    fitted_scale = float(json.loads(STRATEGY_B.read_text())["normalization_factor"])
    print(f"\nScaling factors (read from the established normalisation, not recomputed)")
    print(f"  0-SPION reference (primary) : {reference_factor:.4f}")
    print(f"  Strategy-B fitted scale     : {fitted_scale:.4f}  "
          f"({100*(fitted_scale/reference_factor-1):+.2f}%)")

    # ---- f(rs) from the training samples only ----------------------------
    parts = []
    for sample in TRAINING:
        for seg_dir in sorted((NORMALIZED_SEG / sample).iterdir()):
            if not seg_dir.is_dir():
                continue
            block = pd.read_csv(seg_dir / "voxels.csv.gz",
                                usecols=["rs_selected", "I0_normalized", "flag_high_rs"])
            parts.append(block[block.flag_high_rs == 0][["rs_selected", "I0_normalized"]])
    train = pd.concat(parts, ignore_index=True)
    del parts
    coefficients = np.polyfit(train.rs_selected.to_numpy(np.float64),
                              train.I0_normalized.to_numpy(np.float64), 2)
    f = np.poly1d(coefficients)
    matches = bool(np.allclose(coefficients, PUBLISHED_F, rtol=1e-3))
    print(f"\nf(rs) refitted from {TRAINING} only ({len(train):,} voxels, Pill2 excluded)")
    print(f"  {coefficients[0]:+.4f}*rs^2 {coefficients[1]:+.4f}*rs {coefficients[2]:+.4f}")
    print(f"  matches the published coefficients: {matches}")
    del train

    # ---- apply -----------------------------------------------------------
    i0_norm = np.zeros(shape, dtype=np.float32)
    i0_norm[valid] = i0[valid] / reference_factor
    i0_res = np.zeros(shape, dtype=np.float32)
    i0_res[valid] = i0_norm[valid] - f(rs[valid])

    i0_norm_alt = np.zeros(shape, dtype=np.float32)
    i0_norm_alt[valid] = i0[valid] / fitted_scale
    i0_res_alt = np.zeros(shape, dtype=np.float32)
    i0_res_alt[valid] = i0_norm_alt[valid] - f(rs[valid])

    arrays = {
        "rs_selected": rs, "I0_selected": i0, "B_selected": b_sel,
        "I0_normalized": i0_norm, "I0_residual": i0_res,
        "I0_normalized_strategyB_scale": i0_norm_alt,
        "I0_residual_strategyB_scale": i0_res_alt,
        "fit_success": valid.astype(np.uint8), "quality_ok": quality_ok.astype(np.uint8),
        "flag_high_rs": flag_high_rs.astype(np.uint8),
        "flag_low_information": flag_low_info.astype(np.uint8),
        "n_informative": n_informative,
    }
    for name, volume in arrays.items():
        np.save(OUT / f"{name}.npy", volume)

    z, y, x = np.nonzero(valid)
    pd.DataFrame({
        "x": x.astype(np.int32), "y": y.astype(np.int32), "z": z.astype(np.int32),
        "rs_selected": rs[valid], "I0_selected": i0[valid], "B_selected": b_sel[valid],
        "I0_normalized": i0_norm[valid], "I0_residual": i0_res[valid],
        "I0_normalized_strategyB_scale": i0_norm_alt[valid],
        "I0_residual_strategyB_scale": i0_res_alt[valid],
        "n_informative": n_informative[valid],
        "flag_high_rs": flag_high_rs[valid].astype(np.int8),
        "flag_low_information": flag_low_info[valid].astype(np.int8),
        "quality_ok": quality_ok[valid].astype(np.int8),
    }).to_csv(OUT / "voxel_features.csv.gz", index=False, float_format="%.6g",
              compression="gzip")

    (OUT / "metadata.json").write_text(json.dumps({
        "sample": "Pill2", "source": "complete volume, held-out test set",
        "shape": list(shape), "shape_order": ["z", "y", "x"],
        "axis_map": {"x": "column (i)", "y": "row (j)", "z": "slice (k)"},
        "voxel_spacing_zyx_mm": meta["voxel_spacing_zyx_mm"],
        "n_valid_voxels": int(valid.sum()),
        "normalization": {
            "pipeline": "I0_selected / factor = I0_normalized; I0_normalized - f(rs) = I0_residual",
            "factor_primary": reference_factor,
            "factor_primary_source": ("Pill2's own 0-SPION segmentation median "
                                      "(strategy_A metadata); an unsupervised intensity "
                                      "reference that uses no concentration labels"),
            "factor_alternative": fitted_scale,
            "factor_alternative_source": ("Strategy-B two-way log fit; note this fit "
                                          "included Pill2's labelled concentrations, which "
                                          "is why it is not the primary choice here"),
            "factor_difference_percent": 100*(fitted_scale/reference_factor - 1),
            "f_rs": {"form": "degree-2 polynomial in rs_selected",
                     "coefficients_high_to_low": [float(c) for c in coefficients],
                     "fitted_on": list(TRAINING),
                     "pill2_excluded": True,
                     "matches_published": matches},
        },
        "quality_flags": meta["quality_flags"],
        "provenance": {"refitted": False, "fits_from": str(SOURCE),
                       "originals_modified": False},
        "not_done_yet": ["classification", "any use of Pill2 concentration labels"],
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))

    # ---- report ----------------------------------------------------------
    n = int(valid.sum())
    print("\n" + "=" * 68)
    print(f"Pill2 complete volume — {n:,} valid voxels "
          f"({100*n/np.prod(shape):.1f}% of {int(np.prod(shape)):,})")
    print("=" * 68)
    print(f"\nQuality criteria (carried over unchanged)")
    print(f"  flag_high_rs (rs > 0.15)          {int(flag_high_rs.sum()):>8,}  "
          f"({100*flag_high_rs.sum()/n:5.2f}%)")
    print(f"  flag_low_information (< 4 echoes) {int(flag_low_info.sum()):>8,}  "
          f"({100*flag_low_info.sum()/n:5.2f}%)")
    print(f"  failing at least one              "
          f"{int((~quality_ok & valid).sum()):>8,}  "
          f"({100*(~quality_ok & valid).sum()/n:5.2f}%)")
    print(f"  passing both                      {int(quality_ok.sum()):>8,}  "
          f"({100*quality_ok.sum()/n:5.2f}%)")

    print(f"\n{'feature':32s}{'median':>11s}{'q1':>11s}{'q3':>11s}{'min':>11s}{'max':>11s}")
    for name in ("rs_selected", "I0_selected", "B_selected", "I0_normalized",
                 "I0_residual", "I0_normalized_strategyB_scale",
                 "I0_residual_strategyB_scale"):
        v = arrays[name][valid].astype(np.float64)
        print(f"{name:32s}{np.median(v):11.4f}{np.percentile(v,25):11.4f}"
              f"{np.percentile(v,75):11.4f}{v.min():11.4f}{v.max():11.4f}")

    size = sum(p.stat().st_size for p in OUT.iterdir())
    print(f"\nWritten to {OUT}")
    print(f"  {len(list(OUT.iterdir()))} files, {size/1e6:.0f} MB")
    print(f"elapsed {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
