#!/usr/bin/env python3
"""Complete the Pill2 segmentation-foreground features and re-score the trained models.

Fits only the 42,685 segmented Pill2 voxels that the volume-level Otsu foreground
excluded, using the identical fitting and AIC-selection procedure, then applies the
existing label-light normalisation (factor 2042.62) and the existing f(rs).

The classifiers are loaded from disk and used unchanged. Nothing is retrained,
no Pill2 label enters any fit, and every output goes to new directories.
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
VOL_NORM = MRI / "i0_rs" / "Volume_for_testing" / "Pill2_normalized"     # existing features
SEG_LABELS = MRI / "Normalized_data_i0_rs" / "Segmentations" / "Pill2"    # coords + labels
DICOM = MRI / "Volumes"
MODELS = MRI / "Training"
OUT_FEAT = MRI / "i0_rs" / "Volume_for_testing" / "Pill2_segmentation_complete"
OUT_EVAL = MRI / "Training" / "pill2_complete_segmentation_eval"

FACTOR = 2042.6199951171875
# Read the coefficients at full precision from the existing normalisation metadata.
# The published rounded form (-134.0147, 15.6053, 0.9382) leaves a ~3e-5 offset,
# which would make the newly fitted voxels marginally inconsistent with the
# existing ones.
_NORM_META = json.loads(
    (DATA_ROOT / "i0_rs" / "Volume_for_testing" / "Pill2_normalized"
     / "metadata.json").read_text())
F_COEFFS = tuple(_NORM_META["normalization"]["f_rs"]["coefficients_high_to_low"])
assert [round(c, 4) for c in F_COEFFS] == [-134.0147, 15.6053, 0.9382], F_COEFFS
HIGH_RS, MIN_INFORMATIVE, NOISE_FACTOR = 0.15, 4, 2.0
CLASSES = ["0", "0.05", "0.1", "0.2", "0.25", "0.3", "tissue"]
WORKERS = 9


def main() -> int:
    from joblib import load
    from sklearn.metrics import (balanced_accuracy_score, classification_report,
                                 confusion_matrix, f1_score)
    from echoviewer import EchoSeries
    from echoviewer.diagnostics import LABEL_COLORS, _figure, _save
    from echoviewer.fitting import fit_signals

    started = time.time()
    OUT_FEAT.mkdir(parents=True, exist_ok=True)
    OUT_EVAL.mkdir(parents=True, exist_ok=True)
    f = np.poly1d(F_COEFFS)

    # ---- who needs fitting ----------------------------------------------
    existing_valid = np.load(VOL_NORM / "fit_success.npy") > 0
    shape = existing_valid.shape
    labels = pd.concat(
        [pd.read_csv(p / "voxels.csv.gz", usecols=["x", "y", "z"]).assign(
            material=str(p.name.split("_", 2)[2]), segmentation=p.name)
         for p in sorted(q for q in SEG_LABELS.iterdir() if q.is_dir())], ignore_index=True)
    z = labels.z.to_numpy(np.intp); y = labels.y.to_numpy(np.intp); x = labels.x.to_numpy(np.intp)
    already = existing_valid[z, y, x]
    print(f"segmentation foreground {len(labels):,} voxels: "
          f"{int(already.sum()):,} already fitted, {int((~already).sum()):,} to fit")

    # ---- fit the missing voxels, identical procedure ---------------------
    series = EchoSeries.from_sample(DICOM, "Pill2", cache_size=2)
    try:
        echo_times = [e.value for e in series.echotimes]
        need = ~already
        zn, yn, xn = z[need], y[need], x[need]
        signals = np.zeros((int(need.sum()), len(series)), dtype=np.float32)
        for column in range(len(series)):
            signals[:, column] = series.volume(column).data[zn, yn, xn]
    finally:
        series.close()

    print(f"fitting {signals.shape[0]:,} voxels x {signals.shape[1]} echo times "
          f"on {WORKERS} processes")
    results = fit_signals(echo_times, signals, workers=WORKERS)

    floor = np.median(signals[:, -5:], axis=1, keepdims=True)
    n_inf_new = (signals > NOISE_FACTOR * np.maximum(floor, 1e-6)).sum(axis=1)
    del signals

    chose_a = results["model_selected"] == 1
    new = pd.DataFrame({
        "z": zn, "y": yn, "x": xn,
        "model_selected": results["model_selected"].astype(np.int8),
        "I0_selected": np.where(chose_a, results["I0_noB"], results["I0_B"]),
        "rs_selected": np.where(chose_a, results["rs_noB"], results["rs_B"]),
        "B_selected": np.where(chose_a, 0.0, results["B"]),
        "fit_success": results["fit_success"].astype(np.int8),
        "n_informative": n_inf_new.astype(np.int16),
    })
    print(f"  fit failures among the new voxels: {int((new.fit_success == 0).sum()):,}")

    # ---- assemble the complete set ---------------------------------------
    maps = {n: np.load(VOL_NORM / f"{n}.npy") for n in
            ("rs_selected", "I0_selected", "B_selected", "I0_normalized",
             "I0_residual", "model_selected", "n_informative")}
    zo, yo, xo = z[already], y[already], x[already]
    old = pd.DataFrame({"z": zo, "y": yo, "x": xo,
                        **{n: maps[n][zo, yo, xo] for n in maps},
                        "fit_success": 1})
    old["origin"] = "existing"
    new["I0_normalized"] = new.I0_selected / FACTOR
    new["I0_residual"] = new.I0_normalized - f(new.rs_selected)
    new["origin"] = "newly_fitted"

    complete = pd.concat([old, new], ignore_index=True)
    complete = complete.merge(labels, on=["z", "y", "x"], how="left")
    complete["flag_high_rs"] = (complete.rs_selected > HIGH_RS).astype(np.int8)
    complete["flag_low_information"] = (complete.n_informative < MIN_INFORMATIVE).astype(np.int8)
    complete["quality_ok"] = ((complete.flag_high_rs == 0) &
                              (complete.flag_low_information == 0)).astype(np.int8)
    print(f"complete set: {len(complete):,} voxels "
          f"({int((complete.origin=='existing').sum()):,} existing + "
          f"{int((complete.origin=='newly_fitted').sum()):,} new)")

    # ---- validation -------------------------------------------------------
    print("\nVALIDATION")
    checks = {}
    checks["all_segmented_present"] = len(complete) == len(labels)
    feats = ["rs_selected", "I0_selected", "B_selected", "I0_normalized", "I0_residual"]
    finite = np.isfinite(complete[feats]).all(axis=1)
    checks["all_features_finite"] = bool(finite.all())
    print(f"  all {len(labels):,} segmented voxels present: {checks['all_segmented_present']}")
    print(f"  all features finite: {checks['all_features_finite']}  "
          f"(non-finite {int((~finite).sum())})")

    ref = complete[complete.origin == "existing"]
    same = {c: bool(np.allclose(ref[c], maps[c][ref.z, ref.y, ref.x], rtol=1e-6, atol=1e-9))
            for c in ("rs_selected", "I0_selected", "B_selected", "I0_normalized", "I0_residual")}
    checks["existing_unchanged"] = all(same.values())
    print(f"  existing {len(ref):,} voxels unchanged: {same}")
    recomputed = ref.I0_selected / FACTOR
    checks["factor_unchanged"] = bool(np.allclose(recomputed, ref.I0_normalized, rtol=1e-5))
    checks["f_unchanged"] = bool(np.allclose(ref.I0_normalized - f(ref.rs_selected),
                                             ref.I0_residual, rtol=1e-9, atol=1e-9))
    print(f"  normalisation factor {FACTOR:.4f} reproduces I0_normalized: {checks['factor_unchanged']}")
    print(f"  f(rs) {F_COEFFS} reproduces I0_residual: {checks['f_unchanged']}")
    print(f"  Pill2 labels used in fitting or feature construction: NO "
          f"(labels merged only after all features were built)")

    nf = complete[complete.origin == "newly_fitted"]
    print(f"\n  newly fitted: {len(nf):,}   failures {int((nf.fit_success==0).sum()):,}")
    print(f"    flag_high_rs {int(nf.flag_high_rs.sum()):,} ({100*nf.flag_high_rs.mean():.2f}%)   "
          f"flag_low_information {int(nf.flag_low_information.sum()):,} "
          f"({100*nf.flag_low_information.mean():.2f}%)   "
          f"quality_ok {int(nf.quality_ok.sum()):,} ({100*nf.quality_ok.mean():.2f}%)")
    print(f"    median rs {nf.rs_selected.median():.5f}  median I0 {nf.I0_selected.median():.1f}  "
          f"model A {int((nf.model_selected==1).sum()):,} / B {int((nf.model_selected==2).sum()):,}")
    print("\n  quality flags by material (complete set):")
    for mat, grp in complete.groupby("material"):
        print(f"    {str(mat):8s} n={len(grp):>7,}  high_rs {int(grp.flag_high_rs.sum()):>6,}  "
              f"low_info {int(grp.flag_low_information.sum()):>6,}  "
              f"quality_ok {100*grp.quality_ok.mean():5.1f}%")

    complete.to_csv(OUT_FEAT / "voxels_complete.csv.gz", index=False,
                    float_format="%.6g", compression="gzip")
    for name in feats + ["model_selected", "n_informative", "quality_ok",
                         "flag_high_rs", "flag_low_information"]:
        dtype = np.float32 if name in feats else np.int16
        vol = np.zeros(shape, dtype=dtype)
        vol[complete.z, complete.y, complete.x] = complete[name].to_numpy(dtype)
        np.save(OUT_FEAT / f"{name}.npy", vol)
    (OUT_FEAT / "metadata.json").write_text(json.dumps({
        "sample": "Pill2", "scope": "segmentation foreground (all 384,315 voxels)",
        "shape": list(shape), "shape_order": ["z", "y", "x"],
        "n_voxels": int(len(complete)),
        "n_existing": int((complete.origin == "existing").sum()),
        "n_newly_fitted": int((complete.origin == "newly_fitted").sum()),
        "normalization": {"factor": FACTOR, "f_rs_coefficients_high_to_low": list(F_COEFFS),
                          "pipeline": "I0_selected/factor = I0_normalized; -f(rs) = I0_residual"},
        "fitting": "identical to the existing Pill2 pipeline; AIC selection, ties to Model A",
        "labels_used_for_fitting": False,
        "validation": checks,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2, default=bool))

    # ---- re-score the existing classifiers --------------------------------
    print("\nRE-EVALUATION (classifiers loaded from disk, unchanged)")
    test = complete[complete.material.isin(CLASSES)].reset_index(drop=True)
    print(f"  scoring {len(test):,} voxels, classes {sorted(test.material.unique())}")
    feature_sets = {"rs": ["rs_selected"], "rs+I0_residual": ["rs_selected", "I0_residual"]}
    summary, results_by_model = [], {}
    for name, cols in feature_sets.items():
        model = load(MODELS / f"model_{name.replace('+','_')}.joblib")
        pred = model.predict(test[cols].to_numpy(np.float64))
        present = sorted(set(test.material) | set(pred))
        cm = confusion_matrix(test.material, pred, labels=present)
        rep = classification_report(test.material, pred, labels=present,
                                    output_dict=True, zero_division=0)
        bacc = balanced_accuracy_score(test.material, pred)
        mf1 = f1_score(test.material, pred, average="macro", zero_division=0)
        results_by_model[name] = {"pred": pred, "cm": cm, "labels": present,
                                  "bacc": bacc, "mf1": mf1}
        summary.append({"feature_set": name, "n_voxels": len(test),
                        "balanced_accuracy": bacc, "macro_f1": mf1,
                        "accuracy": rep["accuracy"]})
        print(f"  {name:16s} balanced acc {bacc:.3f}   macro F1 {mf1:.3f}   "
              f"accuracy {rep['accuracy']:.3f}")
        tag = name.replace("+", "_")
        pd.DataFrame(cm, index=present, columns=present).to_csv(
            OUT_EVAL / f"confusion_counts_{tag}.csv")
        pd.DataFrame(cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100,
                     index=present, columns=present).to_csv(
            OUT_EVAL / f"confusion_percent_{tag}.csv")
        pd.DataFrame(rep).T.to_csv(OUT_EVAL / f"per_class_metrics_{tag}.csv")
    pd.DataFrame(summary).to_csv(OUT_EVAL / "model_comparison.csv", index=False)

    fig = _figure(figsize=(15, 6.5))
    axes = fig.subplots(1, 2)
    for ax, (name, r) in zip(axes, results_by_model.items()):
        cm = r["cm"]; pct = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100
        im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(len(r["labels"]))); ax.set_xticklabels(r["labels"], rotation=45)
        ax.set_yticks(range(len(r["labels"]))); ax.set_yticklabels(r["labels"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i,j]:,}\n{pct[i,j]:.1f}%", ha="center", va="center",
                        fontsize=7, color="white" if pct[i, j] > 50 else "black")
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{name}\nbalanced acc {r['bacc']:.3f}, macro F1 {r['mf1']:.3f}", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"Pill2, complete segmentation foreground ({len(test):,} voxels) — "
                 "same trained classifiers, no retraining", fontsize=13)
    _save(fig, OUT_EVAL / "confusion_matrices_complete.png", [])

    (OUT_EVAL / "summary.json").write_text(json.dumps({
        "scope": "all segmented Pill2 voxels", "n_voxels": int(len(test)),
        "newly_fitted": int((complete.origin == "newly_fitted").sum()),
        "classifiers": "loaded unchanged from Training/*.joblib; no retraining",
        "validation": checks, "results": summary,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2, default=bool))

    print(f"\nfeatures -> {OUT_FEAT}")
    print(f"evaluation -> {OUT_EVAL}")
    print(f"elapsed {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
