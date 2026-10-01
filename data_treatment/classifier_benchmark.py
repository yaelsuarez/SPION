#!/usr/bin/env python3
"""Controlled classifier benchmark on the fixed feature set rs_selected + I0_residual.

Five classifiers, one protocol: the same sampled training voxels, the same seed,
the same StandardScaler preprocessing and the same GroupKFold(5) split on
segmentation_id. Pill2 is untouched until every model has been fitted and the
winner has been chosen on grouped CV alone.

The one documented deviation is the RBF SVM: libsvm is O(n^2), so it trains on a
group-preserving subset of the identical sample (see SVM_MAX_TRAIN). Its
train_n is reported in every output.

Nothing existing is modified. Normalisation, features, training data and the
previous models are read-only inputs.
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
P2_COMPLETE = MRI / "i0_rs" / "Volume_for_testing" / "Pill2_segmentation_complete" / "voxels_complete.csv.gz"
P2_NORM = MRI / "i0_rs" / "Volume_for_testing" / "Pill2_normalized"
OUT = MRI / "Classifiers"
PROB_DIR = OUT / "probability_maps"

FEATURES = ["rs_selected", "I0_residual"]
CLASSES = ["0", "0.05", "0.1", "0.2", "0.25", "0.3", "tissue"]
PER_GROUP, SEED, N_FOLDS = 8000, 42, 5
SVM_MAX_TRAIN = 20_000
#: Chosen before any Pill2 number exists. Ties broken by grouped-CV macro-F1.
SELECTION_METRIC = "cv_balanced_accuracy"
FOCUS = [("0.2", "0.25"), ("tissue", "0.1")]
SLICE_STEP = 10


def build_models():
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    return {
        "logistic_regression": (LogisticRegression(max_iter=2000, multi_class="multinomial"), None),
        "random_forest": (RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                 n_jobs=-1, random_state=SEED), None),
        "gradient_boosting": (HistGradientBoostingClassifier(random_state=SEED), None),
        "rbf_svm": (SVC(kernel="rbf", probability=True, cache_size=1000,
                        random_state=SEED), SVM_MAX_TRAIN),
        "knn": (KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1), None),
    }


def subset_by_group(groups, cap, rng):
    """Indices of a <=cap subset that keeps whole groups intact and class balance."""
    if cap is None or cap >= len(groups):
        return np.arange(len(groups))
    per = max(1, cap // len(np.unique(groups)))
    keep = []
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        keep.append(rng.choice(idx, size=min(per, len(idx)), replace=False))
    return np.sort(np.concatenate(keep))


def scores(y_true, y_pred):
    from sklearn.metrics import balanced_accuracy_score, f1_score
    return (balanced_accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, average="macro", zero_division=0,
                     labels=sorted(set(y_true))))


def main() -> int:
    from joblib import dump
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from echoviewer.diagnostics import _figure, _save

    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    PROB_DIR.mkdir(parents=True, exist_ok=True)
    checks = {}

    # ---- training data: identical draw to the established baseline ---------
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
            rows.append(picked)
            del block
    train = pd.concat(rows, ignore_index=True); del rows
    invalid = ~np.isfinite(train[FEATURES]).all(axis=1)
    train = train[~invalid].reset_index(drop=True)
    X = train[FEATURES].to_numpy(np.float64)
    y = train.material.astype(str).to_numpy()
    groups = train.segmentation_id.to_numpy()
    checks["training_source"] = "Syringes + 3D + Pill1 only"
    checks["pill2_in_training"] = False
    checks["n_training"] = int(len(train))
    checks["n_groups"] = int(len(np.unique(groups)))
    checks["non_finite_dropped"] = int(invalid.sum())
    print(f"training {len(train):,} voxels, {checks['n_groups']} groups, seed {SEED}")
    print("class -> segmentations:")
    for c in CLASSES:
        n = train[train.material == c].segmentation_id.nunique()
        print(f"  {c:7s} {n} segmentation(s), {int((y == c).sum()):,} voxels"
              + ("   <- single group: CV cannot validate it" if n < 2 else ""))

    # ---- grouped CV, identical folds for every classifier ------------------
    folds = list(GroupKFold(n_splits=N_FOLDS).split(X, y, groups))
    leak = [len(set(groups[a]) & set(groups[b])) for a, b in folds]
    checks["group_leakage_between_folds"] = int(sum(leak))
    assert sum(leak) == 0, "a segmentation appeared in both train and test of a fold"
    print(f"\nGroupKFold({N_FOLDS}): no segmentation shared between train and test in any fold")

    models = build_models()
    cv_rows, results, fitted = [], {}, {}
    for name, (estimator, cap) in models.items():
        t0 = time.time()
        fold_scores = []
        for k, (tr, te) in enumerate(folds):
            sub = subset_by_group(groups[tr], cap, np.random.default_rng(SEED + k))
            tr = tr[sub]
            pipe = Pipeline([("scale", StandardScaler()),
                             ("clf", type(estimator)(**estimator.get_params()))])
            pipe.fit(X[tr], y[tr])
            pred = pipe.predict(X[te])
            if len(set(y[te])) < 2:
                cv_rows.append({"classifier": name, "fold": k, "scorable": False,
                                "n_train": len(tr), "n_test": len(te)})
                continue
            ba, mf1 = scores(y[te], pred)
            fold_scores.append((ba, mf1))
            cv_rows.append({"classifier": name, "fold": k, "scorable": True,
                            "n_train": len(tr), "n_test": len(te),
                            "test_groups": ", ".join(sorted(set(groups[te]))),
                            "balanced_accuracy": ba, "macro_f1": mf1})
        ba = float(np.mean([s[0] for s in fold_scores]))
        mf1 = float(np.mean([s[1] for s in fold_scores]))

        sub = subset_by_group(groups, cap, np.random.default_rng(SEED))
        final = Pipeline([("scale", StandardScaler()),
                          ("clf", type(estimator)(**estimator.get_params()))])
        final.fit(X[sub], y[sub])
        fitted[name] = final
        results[name] = {"classifier": name, "train_n": int(len(sub)),
                         "cv_balanced_accuracy": ba, "cv_macro_f1": mf1,
                         "cv_folds_scorable": len(fold_scores), "cv_folds": N_FOLDS,
                         "fit_seconds": round(time.time() - t0, 1)}
        print(f"  {name:20s} CV bal acc {ba:.3f}  macro F1 {mf1:.3f}  "
              f"({len(fold_scores)}/{N_FOLDS} folds, train n={len(sub):,}, "
              f"{time.time()-t0:.0f}s)")
    pd.DataFrame(cv_rows).round(4).to_csv(OUT / "cross_validation_folds.csv", index=False)

    best = max(results, key=lambda n: (results[n][SELECTION_METRIC], results[n]["cv_macro_f1"]))
    print(f"\nSELECTED on {SELECTION_METRIC} (Pill2 still untouched): {best}")
    checks["model_selected_before_pill2"] = True
    checks["selection_metric"] = SELECTION_METRIC
    checks["selected_model"] = best

    # ---- Pill2, held out until now -----------------------------------------
    seg = pd.read_csv(P2_COMPLETE)
    seg["material"] = seg.material.astype(str)
    seg = seg[seg.material.isin(CLASSES) & np.isfinite(seg[FEATURES]).all(axis=1)].reset_index(drop=True)
    Xs = seg[FEATURES].to_numpy(np.float64)

    vol = {f: np.load(P2_NORM / f"{f}.npy") for f in FEATURES}
    valid = np.load(P2_NORM / "fit_success.npy") > 0
    for f in FEATURES:
        valid &= np.isfinite(vol[f])
    vz, vy, vx = np.nonzero(valid)
    Xv = np.column_stack([vol[f][valid] for f in FEATURES]).astype(np.float64)
    shape = valid.shape

    # preprocessing equivalence: same features, same maps, same scaler object
    common = seg[(seg.origin == "existing")].head(20000)
    cz = common.z.to_numpy(np.intp); cy = common.y.to_numpy(np.intp); cx = common.x.to_numpy(np.intp)
    dev = max(float(np.nanmax(np.abs(vol[f][cz, cy, cx] - common[f].to_numpy())))
              for f in FEATURES)
    checks["segmented_vs_fullvolume_feature_max_abs_diff"] = dev
    checks["identical_features_all_classifiers"] = FEATURES
    checks["identical_preprocessing"] = "StandardScaler fitted inside each pipeline on the same rows"
    print(f"\nPill2 segmented {len(seg):,} voxels | full volume {len(Xv):,} valid voxels")
    print(f"  feature agreement segmented vs full volume: max abs diff {dev:.3e}")

    order = None
    montage_slices = None
    for name in results:
        model = fitted[name]
        classes_ = list(model.classes_)
        order = classes_ if order is None else order
        assert classes_ == order, "class order differs between classifiers"
        d = OUT / name
        d.mkdir(exist_ok=True)
        dump(model, d / "model.joblib")

        proba = model.predict_proba(Xs)
        pred = np.asarray(classes_)[proba.argmax(1)]
        s = float(np.abs(proba.sum(1) - 1).max())
        results[name]["max_prob_sum_deviation"] = s
        assert s < 1e-6 and proba.min() >= 0, f"{name} produced invalid probabilities"

        ba, mf1 = scores(seg.material.to_numpy(), pred)
        results[name].update({"pill2_balanced_accuracy": ba, "pill2_macro_f1": mf1,
                              "pill2_n": int(len(seg))})
        labels = sorted(set(seg.material) | set(pred), key=CLASSES.index)
        cm = confusion_matrix(seg.material, pred, labels=labels)
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(d / "confusion_counts.csv")
        pct = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100
        pd.DataFrame(pct, index=labels, columns=labels).round(2).to_csv(d / "confusion_percent.csv")
        rep = classification_report(seg.material, pred, labels=labels,
                                    output_dict=True, zero_division=0)
        pd.DataFrame(rep).T.round(4).to_csv(d / "classification_report.csv")
        for a, b in FOCUS:
            if a in labels and b in labels:
                results[name][f"{a}_as_{b}_pct"] = float(pct[labels.index(a), labels.index(b)])
                results[name][f"{a}_recall"] = float(rep[a]["recall"])

        out = seg[["z", "y", "x", "material"]].copy()
        out["predicted"] = pred
        for i, c in enumerate(classes_):
            out[f"p_{c}"] = proba[:, i].astype(np.float32)
        out.to_csv(d / "pill2_segmented_predictions.csv.gz", index=False,
                   float_format="%.5g", compression="gzip")

        pv = model.predict_proba(Xv)
        assert float(np.abs(pv.sum(1) - 1).max()) < 1e-6, f"{name} full-volume probabilities invalid"
        pred_vol = np.zeros(shape, np.int8)
        pred_vol[vz, vy, vx] = pv.argmax(1) + 1  # 0 = outside the fitted volume
        np.save(d / "pill2_fullvolume_prediction.npy", pred_vol)
        fv = pd.DataFrame({"z": vz.astype(np.int16), "y": vy.astype(np.int16),
                           "x": vx.astype(np.int16),
                           "predicted": np.asarray(classes_)[pv.argmax(1)]})
        for i, c in enumerate(classes_):
            fv[f"p_{c}"] = pv[:, i].astype(np.float32)
        fv.to_csv(d / "pill2_fullvolume_predictions.csv.gz", index=False,
                  float_format="%.5g", compression="gzip")
        (d / "metrics.json").write_text(json.dumps(
            {**results[name], "classes": classes_, "features": FEATURES,
             "label_map_fullvolume": {str(i + 1): c for i, c in enumerate(classes_)}},
            indent=2, default=float))

        if montage_slices is None:
            ys = np.flatnonzero(valid.sum(axis=(0, 2)) > 0)
            montage_slices = list(range(int(ys.min()), int(ys.max()) + 1, SLICE_STEP))
        pmaps = np.full((len(classes_),) + shape, np.nan, np.float32)
        for i in range(len(classes_)):
            pmaps[i][vz, vy, vx] = pv[:, i]
        fig = _figure(figsize=(len(montage_slices) * 0.85, len(classes_) * 0.95))
        axes = fig.subplots(len(classes_), len(montage_slices), squeeze=False)
        for r, c in enumerate(classes_):
            for col, sl in enumerate(montage_slices):
                ax = axes[r][col]
                ax.imshow(np.flipud(pmaps[r][:, sl, :]), cmap="magma", vmin=0, vmax=1,
                          interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"y={sl}", fontsize=5)
                if col == 0:
                    ax.set_ylabel(c, fontsize=6)
        fig.suptitle(f"{name} — Pill2 class probability, coronal every {SLICE_STEP} slices, "
                     f"identical positions and scale (0–1) for every classifier", fontsize=9)
        _save(fig, PROB_DIR / f"{name}.svg", [])
        del pv, pmaps, proba
        print(f"  {name:20s} Pill2 bal acc {ba:.3f}  macro F1 {mf1:.3f}")

    # ---- side-by-side summary ---------------------------------------------
    sm = pd.DataFrame(results.values())
    sm["selected_by_cv"] = sm.classifier == best
    cols = ["classifier", "train_n", "cv_balanced_accuracy", "cv_macro_f1",
            "cv_folds_scorable", "pill2_balanced_accuracy", "pill2_macro_f1",
            "0.2_recall", "0.2_as_0.25_pct", "tissue_recall", "tissue_as_0.1_pct",
            "max_prob_sum_deviation", "fit_seconds", "selected_by_cv"]
    sm = sm[[c for c in cols if c in sm.columns]]
    sm.round(4).to_csv(OUT / "summary.csv", index=False)
    print("\n" + sm.round(3).to_string(index=False))

    # combined comparison montage: one representative slice, every class
    ref = int(np.argmax(valid.sum(axis=(0, 2))))
    fig = _figure(figsize=(len(order) * 1.7, len(results) * 1.9))
    axes = fig.subplots(len(results), len(order), squeeze=False)
    for r, name in enumerate(results):
        fv = pd.read_csv(OUT / name / "pill2_fullvolume_predictions.csv.gz")
        m = fv.y.to_numpy() == ref
        zz = fv.z.to_numpy()[m]; xx = fv.x.to_numpy()[m]
        for c, cls in enumerate(order):
            img = np.full((shape[0], shape[2]), np.nan, np.float32)
            img[zz, xx] = fv[f"p_{cls}"].to_numpy()[m]
            ax = axes[r][c]
            ax.imshow(np.flipud(img), cmap="magma", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cls, fontsize=9)
            if c == 0:
                ax.set_ylabel(name, fontsize=8)
        del fv
    fig.suptitle(f"Pill2 class probability, coronal slice y={ref}, identical scale 0–1", fontsize=12)
    _save(fig, PROB_DIR / "combined_comparison.png", [])

    checks["all_probabilities_valid"] = bool(
        max(r["max_prob_sum_deviation"] for r in results.values()) < 1e-6)
    checks["identical_training_sample_and_seed"] = True
    checks["svm_deviation"] = (f"rbf_svm trained on {SVM_MAX_TRAIN:,} group-preserving rows "
                               "of the identical sample; libsvm is O(n^2)")
    (OUT / "critical_checks.json").write_text(json.dumps(checks, indent=2, default=float))
    print("\nCRITICAL CHECKS")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    print(f"\nsaved to {OUT}\nelapsed {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
