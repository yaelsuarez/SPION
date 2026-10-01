#!/usr/bin/env python3
"""Apply the verified label-light normalisation to training data and Pill2.

    I0_selected / reference  ->  I0_normalized  ->  - f(rs)  ->  I0_residual

References (one per sample, each the median I0_selected of that sample's own
0-SPION segmentation, over quality-passing voxels):

    Syringes 7301.30    3D 1069.995    Pill2 2042.62

Pill1 has no 0-SPION segmentation. Its reference is transferred: the median of
its 0.2 segmentation divided by the 0.2 concentration effect learned from
Syringes and 3D alone. No Pill2 data enters that derivation.

f(rs) is fitted on the renormalised Syringes + 3D + Pill1 only and applied
unchanged to Pill2. Pill2's concentration labels are used nowhere; only the
identity of its 0-SPION region, as an intensity reference.

Nothing is refitted. Original fit outputs are untouched; superseded copies of
previously written Pill2 arrays are moved aside rather than deleted.
"""

import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MRI = DATA_ROOT
FITS = MRI / "i0_rs" / "Segmentations"                       # authoritative fit output
FLAGS = MRI / "Normalized_data_i0_rs" / "Segmentations"       # holds the quality flags
VOL_IN = MRI / "i0_rs" / "Volume_for_testing" / "Pill2"
OUT_TRAIN = MRI / "Normalized_data_i0_rs" / "Segmentations_training"
OUT_VOL = MRI / "i0_rs" / "Volume_for_testing" / "Pill2_normalized"

TRAINING = ("Syringes", "3D", "Pill1")
REFERENCE_SAMPLES = ("Syringes", "3D")
STATED = {"Syringes": 7301.30, "3D": 1069.995, "Pill2": 2042.62, "Pill1": 1294.19}
COLUMNS = ["x", "y", "z", "rs_selected", "I0_raw", "B_selected", "model_selected",
           "n_informative", "flag_high_rs", "flag_low_information", "quality_ok"]


def segdirs(root: Path, sample: str):
    return [p for p in sorted((root / sample).iterdir()) if p.is_dir()]


def main() -> int:
    started = time.time()

    # ---- load ------------------------------------------------------------
    print("Loading (read-only)")
    data: dict[tuple[str, str], pd.DataFrame] = {}
    for sample in ("Syringes", "3D", "Pill1", "Pill2"):
        for seg in segdirs(FLAGS, sample):
            block = pd.read_csv(seg / "voxels.csv.gz", usecols=COLUMNS)
            data[(sample, seg.name)] = block
        print(f"  {sample:9s} {len(segdirs(FLAGS, sample))} segmentations, "
              f"{sum(len(v) for (s, _), v in data.items() if s == sample):,} voxels")

    # I0_raw here is I0_selected; confirm against the authoritative fit output.
    check_seg = segdirs(FLAGS, "Syringes")[0].name
    fit = pd.read_csv(FITS / "Syringes" / check_seg / "voxel_fit_data.csv")
    fit = fit[fit.fit_success > 0].reset_index(drop=True)
    chose_a = fit.model_selected.to_numpy() == 1
    expected = np.where(chose_a, fit.I0_noB, fit.I0_B)
    ours = data[("Syringes", check_seg)].sort_values(["z", "y", "x"]).I0_raw.to_numpy()
    ref_sorted = pd.DataFrame({"z": fit.z, "y": fit.y, "x": fit.x, "v": expected}
                              ).sort_values(["z", "y", "x"]).v.to_numpy()
    print(f"  I0_selected matches {FITS.name}/Syringes/{check_seg}: "
          f"{bool(np.allclose(ours, ref_sorted, rtol=1e-4))}")

    def material(seg_name: str) -> str:
        return seg_name.split("_", 2)[2]

    def pooled(sample, mat=None, quality=True):
        parts = [v[v.quality_ok == 1] if quality else v
                 for (s, seg), v in data.items()
                 if s == sample and (mat is None or material(seg) == mat)]
        return pd.concat(parts, ignore_index=True) if parts else None

    # ---- references ------------------------------------------------------
    print("\nReferences")
    reference = {}
    for sample in ("Syringes", "3D", "Pill2"):
        zero = pooled(sample, "0")
        reference[sample] = float(np.median(zero.I0_raw))
        print(f"  {sample:9s} {reference[sample]:12.4f}  own 0-SPION median, "
              f"{len(zero):,} quality voxels   (stated {STATED[sample]})")

    effect = {}
    for mat in sorted({material(seg) for (s, seg) in data
                       if s in REFERENCE_SAMPLES and material(seg) not in ("water", "tissue")},
                      key=float):
        ratios = [float(np.median(pooled(s, mat).I0_raw)) / reference[s]
                  for s in REFERENCE_SAMPLES if pooled(s, mat) is not None]
        effect[mat] = float(np.exp(np.mean(np.log(ratios))))
    pill1_mat = next(material(seg) for (s, seg) in data
                     if s == "Pill1" and material(seg) != "tissue")
    pill1_median = float(np.median(pooled("Pill1", pill1_mat).I0_raw))
    reference["Pill1"] = pill1_median / effect[pill1_mat]
    print(f"  {'Pill1':9s} {reference['Pill1']:12.4f}  = median I0({pill1_mat}) "
          f"{pill1_median:.2f} / effect {effect[pill1_mat]:.4f} "
          f"(from {REFERENCE_SAMPLES})   (stated {STATED['Pill1']})")
    for sample, value in reference.items():
        assert abs(value - STATED[sample]) < 0.01 * STATED[sample], sample

    # ---- f(rs) on the renormalised training data -------------------------
    train = pd.concat(
        [v[v.flag_high_rs == 0].assign(I0n=v[v.flag_high_rs == 0].I0_raw / reference[s])
         for (s, _), v in data.items() if s in TRAINING], ignore_index=True)
    coefficients = np.polyfit(train.rs_selected.to_numpy(np.float64),
                              train.I0n.to_numpy(np.float64), 2)
    f = np.poly1d(coefficients)
    predicted = f(train.rs_selected.to_numpy(np.float64))
    r2 = 1 - float(np.sum((train.I0n - predicted) ** 2)) / \
        float(np.sum((train.I0n - train.I0n.mean()) ** 2))
    print(f"\nf(rs) on {TRAINING}: {len(train):,} voxels (flag_high_rs == 0), Pill2 excluded")
    print(f"  {coefficients[0]:+.4f}*rs^2 {coefficients[1]:+.4f}*rs {coefficients[2]:+.4f}"
          f"   R^2 = {r2:.4f}")
    del train

    provenance = {
        "references": {s: float(v) for s, v in reference.items()},
        "reference_derivation": {
            "Syringes": "median I0_selected of its own 0-SPION segmentation (quality_ok voxels)",
            "3D": "median I0_selected of its own 0-SPION segmentation (quality_ok voxels)",
            "Pill2": ("median I0_selected of its own 0-SPION segmentation (quality_ok voxels); "
                      "used purely as an intensity reference, no concentration labels"),
            "Pill1": (f"median I0_selected of its '{pill1_mat}' segmentation divided by the "
                      f"{pill1_mat} concentration effect learned from {REFERENCE_SAMPLES} only"),
        },
        "concentration_effects_from": list(REFERENCE_SAMPLES),
        "concentration_effects": effect,
        "f_rs": {"form": "degree-2 polynomial in rs_selected",
                 "coefficients_high_to_low": [float(c) for c in coefficients],
                 "fitted_on": list(TRAINING), "pill2_excluded": True,
                 "voxel_filter": "flag_high_rs == 0", "r_squared": float(r2)},
        "pill2_labels_used": {"concentration_distribution": False,
                              "labels_other_than_0": False,
                              "identity_of_0_region": True},
        "pipeline": "I0_normalized = I0_selected / reference; I0_residual = I0_normalized - f(rs)",
    }

    # ---- write training segmentations ------------------------------------
    print(f"\nWriting training segmentations -> {OUT_TRAIN}")
    for sample in TRAINING:
        for seg in segdirs(FLAGS, sample):
            block = data[(sample, seg.name)].copy()
            block["I0_normalized"] = (block.I0_raw / reference[sample]).astype(np.float32)
            block["I0_residual"] = (block.I0_normalized - f(block.rs_selected)).astype(np.float32)
            block["normalization_factor"] = np.float32(reference[sample])
            block["sample"] = sample
            block["segmentation"] = seg.name
            block["material"] = material(seg.name)

            folder = OUT_TRAIN / sample / seg.name
            folder.mkdir(parents=True, exist_ok=True)
            order = ["sample", "segmentation", "material", "x", "y", "z",
                     "rs_selected", "I0_selected", "I0_normalized", "I0_residual",
                     "B_selected", "model_selected", "n_informative",
                     "flag_high_rs", "flag_low_information", "quality_ok",
                     "normalization_factor"]
            block = block.rename(columns={"I0_raw": "I0_selected"})
            block[order].to_csv(folder / "voxels.csv.gz", index=False,
                                float_format="%.6g", compression="gzip")

            shape = tuple(json.loads((FITS / sample / seg.name / "metadata.json").read_text())["shape"])
            z = block.z.to_numpy(np.intp); y = block.y.to_numpy(np.intp); x = block.x.to_numpy(np.intp)
            arrays = {}
            for name, dtype in (("rs_selected", np.float32), ("I0_selected", np.float32),
                                ("I0_normalized", np.float32), ("I0_residual", np.float32),
                                ("B_selected", np.float32), ("model_selected", np.uint8),
                                ("n_informative", np.int16), ("quality_ok", np.uint8),
                                ("flag_high_rs", np.uint8), ("flag_low_information", np.uint8)):
                volume = np.zeros(shape, dtype=dtype)
                volume[z, y, x] = block[name].to_numpy(dtype)
                arrays[name] = volume
            np.savez_compressed(folder / "volumes.npz", **arrays)
            del arrays

            (folder / "metadata.json").write_text(json.dumps({
                "sample": sample, "segmentation": seg.name, "material": material(seg.name),
                "shape": list(shape), "shape_order": ["z", "y", "x"],
                "axis_map": {"x": "column (i)", "y": "row (j)", "z": "slice (k)"},
                "n_voxels": int(len(block)),
                "n_quality_ok": int(block.quality_ok.sum()),
                "normalization": {"strategy": "label-light",
                                  "factor": float(reference[sample]), **provenance},
                "role": "training / calibration",
                "source_fits": str(FITS / sample / seg.name),
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }, indent=2))
            del block
        print(f"  {sample}: {len(segdirs(FLAGS, sample))} segmentations")

    # ---- Pill2 full volume ------------------------------------------------
    print(f"\nWriting Pill2 full volume -> {OUT_VOL}")
    OUT_VOL.mkdir(parents=True, exist_ok=True)
    superseded = OUT_VOL / "superseded_strategyB_scale"
    superseded.mkdir(exist_ok=True)
    for name in ("I0_normalized.npy", "I0_residual.npy",
                 "I0_normalized_strategyB_scale.npy", "I0_residual_strategyB_scale.npy",
                 "voxel_features.csv.gz", "metadata.json"):
        old = OUT_VOL / name
        if old.is_file():
            shutil.move(str(old), str(superseded / name))
    print(f"  previous arrays moved to {superseded.name}/ (nothing deleted)")

    valid = np.load(VOL_IN / "fit_success.npy") > 0
    vol = {name: np.load(VOL_IN / f"{name}.npy") for name in
           ("rs_selected", "I0_selected", "B_selected", "model_selected",
            "n_informative", "quality_ok", "flag_high_rs", "flag_low_information")}
    shape = vol["rs_selected"].shape
    factor = reference["Pill2"]

    i0n = np.zeros(shape, np.float32)
    i0n[valid] = vol["I0_selected"][valid] / factor
    res = np.zeros(shape, np.float32)
    res[valid] = i0n[valid] - f(vol["rs_selected"][valid])
    vol["I0_normalized"] = i0n
    vol["I0_residual"] = res
    vol["fit_success"] = valid.astype(np.uint8)
    for name, volume in vol.items():
        np.save(OUT_VOL / f"{name}.npy", volume)

    z, y, x = np.nonzero(valid)
    pd.DataFrame({
        "x": x.astype(np.int32), "y": y.astype(np.int32), "z": z.astype(np.int32),
        "rs_selected": vol["rs_selected"][valid], "I0_selected": vol["I0_selected"][valid],
        "I0_normalized": i0n[valid], "I0_residual": res[valid],
        "B_selected": vol["B_selected"][valid],
        "model_selected": vol["model_selected"][valid],
        "n_informative": vol["n_informative"][valid],
        "flag_high_rs": vol["flag_high_rs"][valid],
        "flag_low_information": vol["flag_low_information"][valid],
        "quality_ok": vol["quality_ok"][valid],
    }).to_csv(OUT_VOL / "voxel_features.csv.gz", index=False, float_format="%.6g",
              compression="gzip")

    (OUT_VOL / "metadata.json").write_text(json.dumps({
        "sample": "Pill2", "role": "held-out test set (labels not used)",
        "source": "complete volume", "shape": list(shape), "shape_order": ["z", "y", "x"],
        "axis_map": {"x": "column (i)", "y": "row (j)", "z": "slice (k)"},
        "n_valid_voxels": int(valid.sum()),
        "normalization": {"strategy": "label-light", "factor": float(factor), **provenance},
        "note": ("previous Strategy-B-scaled arrays preserved under "
                 "superseded_strategyB_scale/"),
        "source_fits": str(VOL_IN), "refitted": False,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
    print(f"  {int(valid.sum()):,} valid voxels, model_selected.npy included: "
          f"{(OUT_VOL/'model_selected.npy').is_file()}")

    # ---- verification -----------------------------------------------------
    print("\n" + "=" * 72)
    print("VERIFICATION")
    print("=" * 72)
    ok = True

    print("\n1. Normalisation factors")
    for sample in ("Syringes", "3D", "Pill1", "Pill2"):
        good = abs(reference[sample] - STATED[sample]) < 0.01 * STATED[sample]
        ok &= good
        print(f"   {sample:9s} {reference[sample]:12.4f}  matches stated "
              f"{STATED[sample]:>10}: {good}")

    print("\n2. Pill2 concentration labels")
    print("   used to derive any factor or scaling relationship: NO")
    print("   only the identity of its 0-SPION region was used, as an intensity reference")

    print("\n3. f(rs)")
    print(f"   coefficients {[round(float(c), 4) for c in coefficients]}")
    print(f"   fitted on {TRAINING}, Pill2 excluded: True")

    print("\n4. Segmented Pill2 vs full-volume Pill2 (overlapping voxels)")
    seg2 = pd.concat([data[("Pill2", s.name)] for s in segdirs(FLAGS, "Pill2")],
                     ignore_index=True)
    zz = seg2.z.to_numpy(np.intp); yy = seg2.y.to_numpy(np.intp); xx = seg2.x.to_numpy(np.intp)
    inside = valid[zz, yy, xx]
    zz, yy, xx, seg2 = zz[inside], yy[inside], xx[inside], seg2[inside]
    seg_norm = seg2.I0_raw.to_numpy(np.float64) / factor
    seg_res = seg_norm - f(seg2.rs_selected.to_numpy(np.float64))
    checks = {
        "rs_selected": np.allclose(seg2.rs_selected, vol["rs_selected"][zz, yy, xx], rtol=1e-4),
        "I0_selected": np.allclose(seg2.I0_raw, vol["I0_selected"][zz, yy, xx], rtol=1e-4),
        "I0_normalized": np.allclose(seg_norm, i0n[zz, yy, xx], rtol=1e-4),
        "I0_residual": np.allclose(seg_res, res[zz, yy, xx], rtol=1e-4, atol=1e-6),
        "B_selected": np.allclose(seg2.B_selected, vol["B_selected"][zz, yy, xx], rtol=1e-4),
    }
    print(f"   {int(inside.sum()):,} overlapping voxels")
    for name, good in checks.items():
        ok &= bool(good)
        print(f"   {name:16s} identical: {bool(good)}")

    print("\n5. I0_residual == I0_normalized - f(rs)")
    good = bool(np.allclose(res[valid], i0n[valid] - f(vol["rs_selected"][valid]),
                            rtol=1e-5, atol=1e-6))
    ok &= good
    print(f"   Pill2 volume: {good}")
    for sample in TRAINING:
        seg = segdirs(OUT_TRAIN, sample)[0]
        d = pd.read_csv(seg / "voxels.csv.gz")
        g = bool(np.allclose(d.I0_residual, d.I0_normalized - f(d.rs_selected),
                             rtol=1e-3, atol=1e-4))
        ok &= g
        print(f"   {sample:9s} ({seg.name}): {g}")

    print("\n6. Same preprocessing definition everywhere")
    definitions = set()
    for sample in TRAINING:
        md = json.loads((segdirs(OUT_TRAIN, sample)[0] / "metadata.json").read_text())
        definitions.add(md["normalization"]["pipeline"])
        definitions.add(tuple(md["normalization"]["f_rs"]["coefficients_high_to_low"]))
    md = json.loads((OUT_VOL / "metadata.json").read_text())
    definitions.add(md["normalization"]["pipeline"])
    definitions.add(tuple(md["normalization"]["f_rs"]["coefficients_high_to_low"]))
    good = len(definitions) == 2   # one pipeline string + one coefficient tuple
    ok &= good
    print(f"   one pipeline definition and one f(rs) across all outputs: {good}")

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    print(f"\nOutputs:\n  {OUT_TRAIN}\n  {OUT_VOL}")
    print(f"elapsed {time.time()-started:.0f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
