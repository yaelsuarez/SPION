#!/usr/bin/env python3
"""Baseline voxel-level concentration classifier, tested on Pill2 as a held-out phantom.

Training  : label-light normalised Syringes + 3D + Pill1 segmentations
Test      : normalised Pill2, first its labelled segmentations, then the full volume
Features  : rs_selected, and rs_selected + I0_residual
Classifier: multinomial logistic regression on standardised features - the simplest
            interpretable baseline; no hyperparameter search.

No Pill2 voxel is used for training, validation, scaling or model selection.
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
TRAIN_DIR = MRI / "Normalized_data_i0_rs" / "Segmentations_training"
PILL2_SEG = MRI / "Normalized_data_i0_rs" / "Segmentations" / "Pill2"      # coords + labels
PILL2_VOL = MRI / "i0_rs" / "Volume_for_testing" / "Pill2_normalized"      # features
OUT = MRI / "Training"

CLASSES = ["0", "0.05", "0.1", "0.2", "0.25", "0.3", "tissue"]
FEATURE_SETS = {"rs": ["rs_selected"], "rs+I0_residual": ["rs_selected", "I0_residual"]}
PER_GROUP = 8000
SEED = 42
N_FOLDS = 5


def main() -> int:
    from joblib import dump
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (balanced_accuracy_score, classification_report,
                                 confusion_matrix, f1_score)
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from echoviewer.diagnostics import LABEL_COLORS, _figure, _save

    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    saved = []
    rng = np.random.default_rng(SEED)

    # ---- training data ---------------------------------------------------
    print("TRAINING DATA")
    rows, sampling = [], []
    for sample in ("Syringes", "3D", "Pill1"):
        for seg in sorted(p for p in (TRAIN_DIR / sample).iterdir() if p.is_dir()):
            material = seg.name.split("_", 2)[2]
            if material not in CLASSES:
                print(f"  skipped {sample}/{seg.name} (material '{material}' not a class)")
                continue
            block = pd.read_csv(seg / "voxels.csv.gz")
            available = len(block)
            take = min(PER_GROUP, available)
            picked = block.iloc[rng.choice(available, size=take, replace=False)].copy()
            # Force the label from the folder name. Read from the CSV, pandas
            # infers "0"/"0.05" as numbers and "tissue" as a string, producing a
            # mixed-type target that sklearn rejects outright.
            picked["material"] = str(material)
            picked["segmentation_id"] = f"{sample}/{seg.name}"
            rows.append(picked)
            sampling.append({"sample": sample, "segmentation": seg.name,
                             "material": material, "available": available,
                             "requested": PER_GROUP, "actual": take})
            del block
    train = pd.concat(rows, ignore_index=True)
    del rows
    sampling = pd.DataFrame(sampling)
    print(sampling.to_string(index=False))
    print(f"  total training voxels: {len(train):,}   groups: {train.segmentation_id.nunique()}")

    # feature validity, reported not silently dropped
    features_all = ["rs_selected", "I0_residual"]
    invalid = ~np.isfinite(train[features_all]).all(axis=1)
    print(f"\n  non-finite feature rows in training: {int(invalid.sum())} "
          f"({100*invalid.mean():.4f}%) -> dropped")
    train = train[~invalid].reset_index(drop=True)
    print(f"  flagged voxels RETAINED in training: high_rs {int(train.flag_high_rs.sum()):,}, "
          f"low_information {int(train.flag_low_information.sum()):,} "
          f"({100*(1-train.quality_ok.mean()):.2f}% of training set)")

    y = train.material.astype(str).to_numpy()
    groups = train.segmentation_id.to_numpy()

    print("\n  class -> number of distinct segmentations")
    coverage = train.groupby("material").segmentation_id.nunique()
    for cls in CLASSES:
        n = int(coverage.get(cls, 0))
        note = "  <- single segmentation: grouped CV cannot validate this class" if n < 2 else ""
        print(f"    {cls:7s} {n} segmentation(s), {int((y==cls).sum()):,} voxels{note}")

    # ---- grouped cross-validation ---------------------------------------
    print("\nGROUPED CROSS-VALIDATION (segmentation_id as group; no Pill2)")
    cv_rows = []
    for name, cols in FEATURE_SETS.items():
        X = train[cols].to_numpy(np.float64)
        splitter = GroupKFold(n_splits=N_FOLDS)
        scores = []
        for fold, (tr, te) in enumerate(splitter.split(X, y, groups), start=1):
            model = Pipeline([("scale", StandardScaler()),
                              ("clf", LogisticRegression(max_iter=2000, multi_class="multinomial"))])
            model.fit(X[tr], y[tr])
            seen = set(np.unique(y[tr]))
            evaluable = np.isin(y[te], list(seen))
            pred = model.predict(X[te])
            # A fold whose held-out segmentations carry only classes the fold
            # never trained on has nothing to score. That happens because 0.05
            # and tissue each come from a single segmentation.
            if evaluable.sum() == 0:
                bacc = mf1 = float("nan")
            else:
                bacc = balanced_accuracy_score(y[te][evaluable], pred[evaluable])
                mf1 = f1_score(y[te][evaluable], pred[evaluable], average="macro",
                               zero_division=0)
                scores.append((bacc, mf1))
            missing = sorted(set(np.unique(y[te])) - seen)
            cv_rows.append({"feature_set": name, "fold": fold,
                            "train_groups": len(set(groups[tr])), "test_groups": len(set(groups[te])),
                            "leakage": len(set(groups[tr]) & set(groups[te])),
                            "balanced_accuracy": bacc, "macro_f1": mf1,
                            "classes_unseen_in_train": ",".join(missing),
                            "test_voxels_unevaluable": int((~evaluable).sum())})
        a = np.array(scores)
        print(f"  {name:16s} balanced acc {a[:,0].mean():.3f} +- {a[:,0].std():.3f}   "
              f"macro F1 {a[:,1].mean():.3f} +- {a[:,1].std():.3f}   "
              f"({len(scores)}/{N_FOLDS} folds scorable)")
    cv = pd.DataFrame(cv_rows)
    leak = int(cv.leakage.sum())
    print(f"  segmentation leakage across all folds: {leak}  (must be 0)")

    # ---- final models on all training data -------------------------------
    models = {}
    for name, cols in FEATURE_SETS.items():
        model = Pipeline([("scale", StandardScaler()),
                          ("clf", LogisticRegression(max_iter=2000, multi_class="multinomial"))])
        model.fit(train[cols].to_numpy(np.float64), y)
        models[name] = model
        dump(model, OUT / f"model_{name.replace('+','_')}.joblib")

    # ---- Pill2 segmented voxels, with labels -----------------------------
    print("\nPILL2 SEGMENTED EVALUATION (labels used only here, never in training)")
    labels = pd.concat(
        [pd.read_csv(p / "voxels.csv.gz", usecols=["x", "y", "z"]).assign(
            material=p.name.split("_", 2)[2], segmentation=p.name)
         for p in sorted(q for q in PILL2_SEG.iterdir() if q.is_dir())], ignore_index=True)
    vol = {n: np.load(PILL2_VOL / f"{n}.npy") for n in
           ("rs_selected", "I0_residual", "I0_normalized", "I0_selected", "B_selected",
            "fit_success", "quality_ok", "flag_high_rs", "flag_low_information")}
    z = labels.z.to_numpy(np.intp); yy = labels.y.to_numpy(np.intp); x = labels.x.to_numpy(np.intp)
    inside = vol["fit_success"][z, yy, x] > 0
    print(f"  {len(labels):,} labelled voxels; {int(inside.sum()):,} inside the volume foreground "
          f"({int((~inside).sum()):,} outside, excluded from this evaluation)")
    test = labels[inside].copy()
    z, yy, x = z[inside], yy[inside], x[inside]
    for n in ("rs_selected", "I0_residual", "quality_ok", "flag_high_rs", "flag_low_information"):
        test[n] = vol[n][z, yy, x]
    test["material"] = test.material.astype(str)
    test = test[test.material.isin(CLASSES)].reset_index(drop=True)
    print(f"  evaluable against the class list: {len(test):,} voxels, "
          f"classes {sorted(test.material.unique())}")

    results = {}
    for name, cols in FEATURE_SETS.items():
        model = models[name]
        Xt = test[cols].to_numpy(np.float64)
        pred = model.predict(Xt)
        proba = model.predict_proba(Xt)
        present = sorted(set(test.material) | set(pred))
        cm = confusion_matrix(test.material, pred, labels=present)
        report = classification_report(test.material, pred, labels=present,
                                       output_dict=True, zero_division=0)
        results[name] = {"pred": pred, "proba": proba, "cm": cm, "labels": present,
                         "report": report,
                         "balanced_accuracy": balanced_accuracy_score(test.material, pred),
                         "macro_f1": f1_score(test.material, pred, average="macro", zero_division=0)}
        print(f"  {name:16s} balanced acc {results[name]['balanced_accuracy']:.3f}   "
              f"macro F1 {results[name]['macro_f1']:.3f}")
        pd.DataFrame(cm, index=present, columns=present).to_csv(
            OUT / f"confusion_counts_{name.replace('+','_')}.csv")
        pd.DataFrame(cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100,
                     index=present, columns=present).to_csv(
            OUT / f"confusion_percent_{name.replace('+','_')}.csv")
        pd.DataFrame(report).T.to_csv(OUT / f"per_class_metrics_{name.replace('+','_')}.csv")

    # ---- figures ---------------------------------------------------------
    fig = _figure(figsize=(15, 6.5))
    axes = fig.subplots(1, 2)
    for ax, (name, r) in zip(axes, results.items()):
        cm = r["cm"]; pct = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100
        im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(len(r["labels"]))); ax.set_xticklabels(r["labels"], rotation=45)
        ax.set_yticks(range(len(r["labels"]))); ax.set_yticklabels(r["labels"])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i,j]:,}\n{pct[i,j]:.1f}%", ha="center", va="center",
                        fontsize=7, color="white" if pct[i, j] > 50 else "black")
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{name}\nbalanced acc {r['balanced_accuracy']:.3f}, "
                     f"macro F1 {r['macro_f1']:.3f}", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Pill2 segmented voxels — confusion (counts and row %)", fontsize=13)
    _save(fig, OUT / "confusion_matrices_pill2.png", saved)

    fig = _figure(figsize=(14, 5.5))
    axes = fig.subplots(1, 2)
    for ax, (name, r) in zip(axes, results.items()):
        conf = r["proba"].max(axis=1)
        for cls in sorted(set(test.material)):
            m = test.material.to_numpy() == cls
            ax.hist(conf[m], bins=50, histtype="step", lw=1.5,
                    color=LABEL_COLORS.get(cls, "#666"), label=f"{cls} (n={m.sum():,})",
                    density=True)
        ax.set_xlabel("max predicted probability"); ax.set_ylabel("density")
        ax.set_title(name); ax.legend(fontsize=7); ax.grid(alpha=.25)
    fig.suptitle("Pill2 prediction confidence by true class", fontsize=12)
    _save(fig, OUT / "prediction_confidence_pill2.png", saved)

    best = "rs+I0_residual"
    shape = vol["rs_selected"].shape
    code = {c: i + 1 for i, c in enumerate(CLASSES)}
    truth_map = np.zeros(shape, np.uint8); pred_map = np.zeros(shape, np.uint8)
    truth_map[test.z, test.y, test.x] = [code[c] for c in test.material]
    pred_map[test.z, test.y, test.x] = [code[c] for c in results[best]["pred"]]
    k = int(np.argmax((truth_map > 0).sum(axis=(1, 2))))
    fig = _figure(figsize=(13, 7))
    axes = fig.subplots(1, 2)
    for ax, (arr, title) in zip(axes, ((truth_map, "ground truth"), (pred_map, f"predicted ({best})"))):
        ax.imshow(np.ma.masked_where(arr[k] == 0, arr[k]), cmap="tab10", vmin=0, vmax=10,
                  interpolation="nearest")
        ax.set_title(f"Pill2 segmented — {title}, axial {k}", fontsize=10); ax.set_axis_off()
    fig.suptitle(f"class codes: {code}", fontsize=9)
    _save(fig, OUT / "pill2_segmented_prediction_vs_truth.png", saved)

    # ---- full volume -----------------------------------------------------
    print("\nFULL-VOLUME PILL2 INFERENCE (no labels, not used for any fitting)")
    valid = vol["fit_success"] > 0
    Xv = np.column_stack([vol[c][valid] for c in FEATURE_SETS[best]])
    finite = np.isfinite(Xv).all(axis=1)
    print(f"  {int(valid.sum()):,} valid voxels; non-finite features: {int((~finite).sum())}")
    pred_full = np.zeros(shape, np.uint8)
    zz, yv, xv = np.nonzero(valid)
    p = models[best].predict(Xv[finite])
    pred_full[zz[finite], yv[finite], xv[finite]] = [code[c] for c in p]
    np.save(OUT / "pill2_fullvolume_prediction.npy", pred_full)
    counts = pd.Series(p).value_counts().reindex(CLASSES).fillna(0).astype(int)
    print("  predicted class counts over the whole volume:")
    for cls in CLASSES:
        print(f"    {cls:7s} {counts[cls]:>9,}  ({100*counts[cls]/len(p):5.2f}%)")

    fig = _figure(figsize=(16, 6))
    axes = fig.subplots(1, 3)
    for ax, (view, idx) in zip(axes, (("axial", k),
                                      ("coronal", int(np.argmax((pred_full > 0).sum(axis=(0, 2))))),
                                      ("sagittal", int(np.argmax((pred_full > 0).sum(axis=(0, 1))))))):
        sl = pred_full[idx] if view == "axial" else (
            np.flipud(pred_full[:, idx, :]) if view == "coronal" else np.flipud(pred_full[:, :, idx]))
        ax.imshow(np.ma.masked_where(sl == 0, sl), cmap="tab10", vmin=0, vmax=10,
                  interpolation="nearest")
        ax.set_title(f"{view} {idx}", fontsize=10); ax.set_axis_off()
    fig.suptitle(f"Pill2 full-volume prediction ({best})   class codes: {code}", fontsize=11)
    _save(fig, OUT / "pill2_fullvolume_prediction.png", saved)

    # ---- comparison + summary --------------------------------------------
    comparison = pd.DataFrame([{
        "feature_set": n,
        "cv_scorable_folds": int(cv[cv.feature_set == n].balanced_accuracy.notna().sum()),
        "cv_balanced_accuracy_mean": cv[cv.feature_set == n].balanced_accuracy.mean(),
        "cv_balanced_accuracy_sd": cv[cv.feature_set == n].balanced_accuracy.std(),
        "cv_macro_f1_mean": cv[cv.feature_set == n].macro_f1.mean(),
        "cv_macro_f1_sd": cv[cv.feature_set == n].macro_f1.std(),
        "pill2_balanced_accuracy": results[n]["balanced_accuracy"],
        "pill2_macro_f1": results[n]["macro_f1"],
    } for n in FEATURE_SETS])
    comparison.to_csv(OUT / "model_comparison.csv", index=False)
    cv.to_csv(OUT / "cross_validation_folds.csv", index=False)
    sampling.to_csv(OUT / "training_sampling.csv", index=False)

    summary = {
        "classifier": "multinomial logistic regression on standardised features",
        "feature_sets": FEATURE_SETS, "classes": CLASSES,
        "sampling": {"per_group_requested": PER_GROUP, "seed": SEED,
                     "total_training_voxels": int(len(train)),
                     "groups": int(train.segmentation_id.nunique())},
        "validation": {"scheme": f"GroupKFold({N_FOLDS}) on segmentation_id",
                       "leakage_groups": leak,
                       "classes_with_one_segmentation": [c for c in CLASSES
                                                         if int(coverage.get(c, 0)) < 2]},
        "normalization": json.loads((PILL2_VOL / "metadata.json").read_text())["normalization"],
        "cv": comparison.to_dict("records"),
        "pill2_segmented": {n: {"balanced_accuracy": results[n]["balanced_accuracy"],
                                "macro_f1": results[n]["macro_f1"],
                                "n_voxels": int(len(test))} for n in FEATURE_SETS},
        "pill2_fullvolume_class_counts": {c: int(counts[c]) for c in CLASSES},
        "pill2_labels_used": "only to score the segmented evaluation; never in training/CV",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nSaved to {OUT}  ({len(list(OUT.iterdir()))} files)")
    print(f"elapsed {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
