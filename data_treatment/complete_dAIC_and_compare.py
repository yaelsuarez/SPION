#!/usr/bin/env python3
"""Recover the missing Pill2 dAIC values and test dAIC as a classifier feature.

The 42,685 Pill2 voxels refitted earlier stored only the AIC-selected parameters,
so dAIC = AIC_B - AIC_A cannot be reconstructed from them. The identical
deterministic fit is re-run to obtain both models' AIC; the resulting parameters
are then verified bit-identical to what is stored, so nothing is changed - only
AIC is added.

Models 1 and 2 are loaded from disk unchanged. Models 3 and 4 are new feature
sets, so they must be fitted; they use the identical protocol, sampling, seed,
preprocessing and training data as the baseline. Pill2 is never used for
training or model selection.
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
FITS = MRI / "i0_rs" / "Segmentations"
TRAIN_DIR = MRI / "Normalized_data_i0_rs" / "Segmentations_training"
P2_VOL = MRI / "i0_rs" / "Volume_for_testing" / "Pill2"
P2_COMPLETE = MRI / "i0_rs" / "Volume_for_testing" / "Pill2_segmentation_complete" / "voxels_complete.csv.gz"
DICOM = MRI / "Volumes"
MODELS = MRI / "Training"
OUT = MRI / "Training" / "diagnosis" / "dAIC"

CLASSES = ["0", "0.05", "0.1", "0.2", "0.25", "0.3", "tissue"]
PER_GROUP, SEED, WORKERS = 8000, 42, 9
FEATURE_SETS = {
    "rs": ["rs_selected"],
    "rs+I0_residual": ["rs_selected", "I0_residual"],
    "rs+dAIC": ["rs_selected", "dAIC"],
    "rs+I0_residual+dAIC": ["rs_selected", "I0_residual", "dAIC"],
}


def main() -> int:
    from joblib import dump, load
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (balanced_accuracy_score, classification_report,
                                 confusion_matrix, f1_score)
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from echoviewer import EchoSeries
    from echoviewer.diagnostics import _figure, _save
    from echoviewer.fitting import fit_signals

    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    complete = pd.read_csv(P2_COMPLETE)
    complete["material"] = complete.material.astype(str)
    missing = complete[complete.origin == "newly_fitted"].reset_index(drop=True)
    print(f"Pill2 complete {len(complete):,}; missing dAIC for {len(missing):,}")

    # ---- verification: does re-running reproduce the stored AIC exactly? ---
    print("\nVERIFY the AIC path on voxels that already have AIC")
    existing_valid = np.load(P2_VOL / "fit_success.npy") > 0
    aic_a_map = np.load(P2_VOL / "AIC_A.npy"); aic_b_map = np.load(P2_VOL / "AIC_B.npy")
    have = complete[complete.origin == "existing"]
    check = have.sample(3000, random_state=SEED)
    series = EchoSeries.from_sample(DICOM, "Pill2", cache_size=2)
    try:
        te = [e.value for e in series.echotimes]
        def curves(frame):
            z = frame.z.to_numpy(np.intp); y = frame.y.to_numpy(np.intp); x = frame.x.to_numpy(np.intp)
            out = np.zeros((len(frame), len(series)), np.float32)
            for c in range(len(series)):
                out[:, c] = series.volume(c).data[z, y, x]
            return out
        sig_check = curves(check)
        sig_missing = curves(missing)
    finally:
        series.close()

    r_check = fit_signals(te, sig_check, workers=WORKERS)
    zc = check.z.to_numpy(np.intp); yc = check.y.to_numpy(np.intp); xc = check.x.to_numpy(np.intp)
    d_a = np.abs(r_check["AIC_noB"] - aic_a_map[zc, yc, xc]).max()
    d_b = np.abs(r_check["AIC_B"] - aic_b_map[zc, yc, xc]).max()
    print(f"  3,000 sampled voxels: max |AIC_A recomputed - stored| = {d_a:.3e}")
    print(f"                        max |AIC_B recomputed - stored| = {d_b:.3e}")
    identical_aic = bool(d_a < 1e-3 and d_b < 1e-3)
    print(f"  -> dAIC is computed identically: {identical_aic}")
    del sig_check, r_check

    # ---- recover AIC for the missing voxels, verifying params unchanged ----
    r_new = fit_signals(te, sig_missing, workers=WORKERS)
    del sig_missing
    chose_a = r_new["model_selected"] == 1
    same = {
        "model_selected": bool(np.array_equal(r_new["model_selected"].astype(np.int8),
                                              missing.model_selected.to_numpy(np.int8))),
        "rs_selected": bool(np.allclose(np.where(chose_a, r_new["rs_noB"], r_new["rs_B"]),
                                        missing.rs_selected, rtol=1e-5, atol=1e-8)),
        "I0_selected": bool(np.allclose(np.where(chose_a, r_new["I0_noB"], r_new["I0_B"]),
                                        missing.I0_selected, rtol=1e-5, atol=1e-4)),
        "B_selected": bool(np.allclose(np.where(chose_a, 0.0, r_new["B"]),
                                       missing.B_selected, rtol=1e-5, atol=1e-4)),
    }
    print(f"\n  re-run reproduces the stored parameters: {same}")
    assert all(same.values()), "re-run changed the stored parameters"
    missing_dAIC = r_new["AIC_B"] - r_new["AIC_noB"]

    # ---- assemble complete dAIC -------------------------------------------
    complete["dAIC"] = np.nan
    ex = complete.origin == "existing"
    ze = complete.loc[ex, "z"].to_numpy(np.intp)
    ye = complete.loc[ex, "y"].to_numpy(np.intp)
    xe = complete.loc[ex, "x"].to_numpy(np.intp)
    complete.loc[ex, "dAIC"] = aic_b_map[ze, ye, xe] - aic_a_map[ze, ye, xe]
    complete.loc[complete.origin == "newly_fitted", "dAIC"] = missing_dAIC
    print(f"  dAIC now present for {int(complete.dAIC.notna().sum()):,} of {len(complete):,} "
          f"({100*complete.dAIC.notna().mean():.1f}%)")
    complete[["z", "y", "x", "material", "origin", "dAIC"]].to_csv(
        OUT / "pill2_dAIC_complete.csv.gz", index=False, float_format="%.6g",
        compression="gzip")

    # ---- training data, identical protocol ---------------------------------
    # The draw order must match the baseline exactly, so sample first with the
    # same rng call sequence and only then attach dAIC to the picked rows.
    rng = np.random.default_rng(SEED)
    rows = []
    for sample in ("Syringes", "3D", "Pill1"):
        for seg in sorted(p for p in (TRAIN_DIR / sample).iterdir() if p.is_dir()):
            material = seg.name.split("_", 2)[2]
            if material not in CLASSES:
                continue
            block = pd.read_csv(seg / "voxels.csv.gz")
            take = min(PER_GROUP, len(block))
            picked = block.iloc[rng.choice(len(block), size=take, replace=False)].copy()
            picked["material"] = str(material)
            picked["segmentation_id"] = f"{sample}/{seg.name}"
            fit = pd.read_csv(FITS / sample / seg.name / "voxel_fit_data.csv",
                              usecols=["x", "y", "z", "AIC_noB", "AIC_B"])
            fit["dAIC"] = fit.AIC_B - fit.AIC_noB
            order = picked.index.to_numpy()
            picked = picked.merge(fit[["x", "y", "z", "dAIC"]], on=["x", "y", "z"], how="left")
            picked.index = order
            rows.append(picked)
            del block, fit
    train = pd.concat(rows, ignore_index=True); del rows
    invalid = ~np.isfinite(train[["rs_selected", "I0_residual", "dAIC"]]).all(axis=1)
    print(f"\nnon-finite feature rows in training: {int(invalid.sum())} "
          f"({100*invalid.mean():.4f}%) -> dropped")
    train = train[~invalid].reset_index(drop=True)
    y = train.material.astype(str).to_numpy()
    print(f"training sample: {len(train):,} voxels, {train.segmentation_id.nunique()} groups, "
          f"seed {SEED}, {PER_GROUP}/group  (baseline protocol)")

    # ---- models: 1-2 loaded unchanged, 3-4 fitted with the same protocol ---
    models = {}
    for name in FEATURE_SETS:
        path = MODELS / f"model_{name.replace('+','_')}.joblib"
        if path.is_file():
            models[name] = load(path)
            print(f"  {name:22s} loaded unchanged from {path.name}")
        else:
            m = Pipeline([("scale", StandardScaler()),
                          ("clf", LogisticRegression(max_iter=2000,
                                                     multi_class="multinomial"))])
            m.fit(train[FEATURE_SETS[name]].to_numpy(np.float64), y)
            models[name] = m
            dump(m, OUT / f"model_{name.replace('+','_')}.joblib")
            print(f"  {name:22s} fitted (new feature set, identical protocol) -> "
                  f"diagnosis/dAIC/")

    # ---- evaluate on all labelled Pill2 voxels -----------------------------
    test = complete[complete.material.isin(CLASSES) & complete.dAIC.notna()].reset_index(drop=True)
    print(f"\nevaluating on {len(test):,} labelled Pill2 voxels")
    summary = []
    cms = {}
    for name, cols in FEATURE_SETS.items():
        pred = models[name].predict(test[cols].to_numpy(np.float64))
        present = sorted(set(test.material) | set(pred))
        cm = confusion_matrix(test.material, pred, labels=present)
        rep = classification_report(test.material, pred, labels=present,
                                    output_dict=True, zero_division=0)
        bacc = balanced_accuracy_score(test.material, pred)
        mf1 = f1_score(test.material, pred, average="macro", zero_division=0)
        cms[name] = (cm, present, bacc, mf1)
        summary.append({"feature_set": name, "n_voxels": len(test),
                        "balanced_accuracy": bacc, "macro_f1": mf1,
                        "accuracy": rep["accuracy"]})
        tag = name.replace("+", "_")
        pd.DataFrame(cm, index=present, columns=present).to_csv(OUT / f"confusion_counts_{tag}.csv")
        pd.DataFrame(cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100,
                     index=present, columns=present).to_csv(OUT / f"confusion_percent_{tag}.csv")
        pd.DataFrame(rep).T.to_csv(OUT / f"per_class_metrics_{tag}.csv")
        print(f"  {name:22s} balanced acc {bacc:.3f}   macro F1 {mf1:.3f}   "
              f"accuracy {rep['accuracy']:.3f}")
    pd.DataFrame(summary).to_csv(OUT / "model_comparison.csv", index=False)

    fig = _figure(figsize=(19, 5.6))
    axes = fig.subplots(1, 4)
    for ax, (name, (cm, present, bacc, mf1)) in zip(axes, cms.items()):
        pct = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100
        im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(len(present))); ax.set_xticklabels(present, rotation=45, fontsize=7)
        ax.set_yticks(range(len(present))); ax.set_yticklabels(present, fontsize=7)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{pct[i,j]:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if pct[i, j] > 50 else "black")
        ax.set_title(f"{name}\nbal acc {bacc:.3f}, macro F1 {mf1:.3f}", fontsize=9)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.suptitle(f"Pill2, all {len(test):,} labelled voxels — does dAIC add value?", fontsize=13)
    _save(fig, OUT / "confusion_matrices.png", [])

    (OUT / "summary.json").write_text(json.dumps({
        "dAIC_verified_identical": identical_aic,
        "max_abs_AIC_A_difference": float(d_a), "max_abs_AIC_B_difference": float(d_b),
        "stored_parameters_unchanged": same,
        "n_dAIC_recovered": int(len(missing)),
        "n_evaluated": int(len(test)),
        "protocol": {"per_group": PER_GROUP, "seed": SEED,
                     "classifier": "multinomial logistic regression on standardised features",
                     "models_1_2": "loaded unchanged", "models_3_4": "new feature sets, same protocol"},
        "results": summary,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2, default=float))
    print(f"\nsaved to {OUT}")
    print(f"elapsed {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
