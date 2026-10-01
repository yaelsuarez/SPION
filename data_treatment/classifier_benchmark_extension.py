#!/usr/bin/env python3
"""Extend the classifier benchmark with LDA, QDA, GMM, hierarchical and ordinal.

Same frozen protocol as the first five: identical sampled training voxels, seed
42, StandardScaler preprocessing, GroupKFold(5) on segmentation_id, and Pill2
held out entirely. Nothing in the existing per-classifier directories is read
for fitting or written to.

Three of the five cannot produce a flat 7-class decision on their own and are
flagged as such in every output:

* GMM is fitted with no labels at all. Training labels are used afterwards, once,
  to map each discovered component to the class that dominates it. A class that
  never dominates a component becomes unpredictable by construction.
* Hierarchical splits the problem into a tissue/SPION gate and a concentration
  model trained only on SPION voxels.
* Ordinal places the six concentrations on their natural order and predicts
  through cumulative P(y > c) models. `tissue` has no position on a
  concentration axis, so it is handled by the same gate as the hierarchical
  model, never by the ordinal part.
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

FEATURES = ["rs_selected", "I0_residual"]
CLASSES = ["0", "0.05", "0.1", "0.2", "0.25", "0.3", "tissue"]
CONCENTRATIONS = ["0", "0.05", "0.1", "0.2", "0.25", "0.3"]  # ascending, ordinal
PER_GROUP, SEED, N_FOLDS = 8000, 42, 5
GMM_GRID = (7, 10, 14, 20, 28)
FOCUS = [("0.2", "0.25"), ("tissue", "0.1")]


def _pipe(estimator):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("scale", StandardScaler()), ("clf", estimator)])


def _lr():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=2000)


class _Constant:
    """Stand-in for a stage whose training fold contains a single class.

    `tissue` and `0.05` each come from one segmentation, so a GroupKFold fold
    that holds one out leaves the gate — or an ordinal step — with nothing to
    discriminate. Failing there would hide a real property of the data, so the
    stage degenerates to the one class it saw and the fold is still scored.
    """

    def __init__(self, value):
        self.classes_ = np.array([value])

    def predict_proba(self, X):
        return np.ones((len(X), 1))


def _fit_stage(X, y):
    # Do not force object dtype here: the ordinal steps pass integer targets and
    # sklearn reads an object array of ints as an unknown (regression) target.
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return _Constant(y[0])
    return _pipe(_lr()).fit(X, y)


def _stage_proba(stage, X, target):
    """P(target) from a stage that may be degenerate."""
    p = stage.predict_proba(X)
    classes = list(stage.classes_)
    return p[:, classes.index(target)] if target in classes else np.zeros(len(X))


class GMMClassifier:
    """Unsupervised mixture; components mapped to classes by training labels only."""

    natural_7_class = False
    note = ("fitted with no labels; components mapped to classes afterwards using "
            "training labels only; classes that dominate no component are unpredictable")

    def __init__(self, grid=GMM_GRID, seed=SEED):
        self.grid, self.seed = grid, seed

    def fit(self, X, y):
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler
        self.scaler_ = StandardScaler().fit(X)
        Z = self.scaler_.transform(X)
        best, self.bic_ = None, {}
        for k in self.grid:                      # selection by BIC: no labels involved
            g = GaussianMixture(n_components=k, covariance_type="full",
                                random_state=self.seed, max_iter=300).fit(Z)
            self.bic_[k] = float(g.bic(Z))
            if best is None or self.bic_[k] < self.bic_[best.n_components]:
                best = g
        self.gmm_ = best
        self.n_components_ = best.n_components
        # the only place labels are touched, and only training labels
        assign = self.gmm_.predict(Z)
        self.classes_ = np.array(sorted(set(y)))
        table = pd.crosstab(assign, y)
        self.component_class_ = {int(c): str(table.loc[c].idxmax()) for c in table.index}
        self.unmapped_ = [c for c in self.classes_
                          if c not in set(self.component_class_.values())]
        return self

    def predict_proba(self, X):
        resp = self.gmm_.predict_proba(self.scaler_.transform(X))
        out = np.zeros((len(X), len(self.classes_)))
        index = {c: i for i, c in enumerate(self.classes_)}
        for comp, cls in self.component_class_.items():
            out[:, index[cls]] += resp[:, comp]
        return out

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class HierarchicalClassifier:
    """Gate tissue vs SPION, then concentration among SPION voxels only."""

    natural_7_class = False
    note = "two stages: tissue/SPION gate, then concentration trained on SPION voxels only"

    def fit(self, X, y):
        y = np.asarray(y, dtype=object)
        self.classes_ = np.array(sorted(set(y)))
        self.gate_ = _fit_stage(X, np.where(y == "tissue", "tissue", "spion"))
        m = y != "tissue"
        self.conc_ = _fit_stage(X[m], y[m])
        return self

    def predict_proba(self, X):
        out = np.zeros((len(X), len(self.classes_)))
        index = {c: i for i, c in enumerate(self.classes_)}
        p_tissue = _stage_proba(self.gate_, X, "tissue")
        if "tissue" in index:
            out[:, index["tissue"]] = p_tissue
        pc = self.conc_.predict_proba(X) * (1.0 - p_tissue)[:, None]
        for i, c in enumerate(self.conc_.classes_):
            out[:, index[c]] = pc[:, i]
        return out

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class OrdinalClassifier:
    """Frank & Hall cumulative decomposition over the ordered concentrations.

    `tissue` is not on the concentration axis and is handled by the same
    tissue/SPION gate as the hierarchical model.
    """

    natural_7_class = False
    note = ("concentrations modelled as ordered via cumulative P(y > c); tissue is off "
            "the concentration axis and comes from a separate tissue/SPION gate")

    def fit(self, X, y):
        y = np.asarray(y, dtype=object)
        self.classes_ = np.array(sorted(set(y)))
        self.order_ = [c for c in CONCENTRATIONS if c in set(y)]
        self.gate_ = _fit_stage(X, np.where(y == "tissue", "tissue", "spion"))
        m = y != "tissue"
        Xs, ys = X[m], y[m]
        rank = {c: i for i, c in enumerate(self.order_)}
        r = np.array([rank[v] for v in ys])
        self.steps_ = []
        for k in range(len(self.order_) - 1):    # P(y > order_[k])
            self.steps_.append(_fit_stage(Xs, (r > k).astype(int)))
        self.violations_ = 0.0
        return self

    def _concentration_proba(self, X):
        n = len(X)
        cum = np.ones((n, len(self.order_) + 1))
        cum[:, -1] = 0.0
        for k, m in enumerate(self.steps_):
            cum[:, k + 1] = _stage_proba(m, X, 1)
        p = cum[:, :-1] - cum[:, 1:]
        self.violations_ = float((p < 0).any(axis=1).mean())
        p = np.clip(p, 0, None)                  # cumulative models are not forced monotone
        return p / np.maximum(p.sum(1, keepdims=True), 1e-12)

    def predict_proba(self, X):
        out = np.zeros((len(X), len(self.classes_)))
        index = {c: i for i, c in enumerate(self.classes_)}
        p_tissue = _stage_proba(self.gate_, X, "tissue")
        if "tissue" in index:
            out[:, index["tissue"]] = p_tissue
        pc = self._concentration_proba(X) * (1.0 - p_tissue)[:, None]
        for i, c in enumerate(self.order_):
            out[:, index[c]] = pc[:, i]
        return out

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


class Standard:
    """Plain multiclass estimator on the shared pipeline. Module level so it pickles."""

    natural_7_class = True
    note = "standard multiclass estimator on the shared pipeline"

    def __init__(self, factory):
        self.factory = factory

    def fit(self, X, y):
        self.model_ = _pipe(self.factory()).fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X)


def build_methods():
    from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                               QuadraticDiscriminantAnalysis)
    return {
        "lda": lambda: Standard(LinearDiscriminantAnalysis),
        "qda": lambda: Standard(QuadraticDiscriminantAnalysis),
        "gmm": GMMClassifier,
        "hierarchical": HierarchicalClassifier,
        "ordinal": OrdinalClassifier,
    }


def main() -> int:
    from joblib import dump
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 classification_report, confusion_matrix, f1_score)
    from sklearn.model_selection import GroupKFold

    started = time.time()
    checks = {"pill2_used_for_fitting_or_mapping": False,
              "features": FEATURES, "seed": SEED, "per_group": PER_GROUP}

    # ---- identical training draw ------------------------------------------
    rng = np.random.default_rng(SEED)
    rows = []
    for sample in ("Syringes", "3D", "Pill1"):
        for seg in sorted(p for p in (TRAIN_DIR / sample).iterdir() if p.is_dir()):
            material = seg.name.split("_", 2)[2]
            if material not in CLASSES:
                continue
            block = pd.read_csv(seg / "voxels.csv.gz")
            picked = block.iloc[rng.choice(len(block), size=min(PER_GROUP, len(block)),
                                           replace=False)].copy()
            picked["material"] = str(material)
            picked["segmentation_id"] = f"{sample}/{seg.name}"
            rows.append(picked)
            del block
    train = pd.concat(rows, ignore_index=True); del rows
    train = train[np.isfinite(train[FEATURES]).all(axis=1)].reset_index(drop=True)
    X = train[FEATURES].to_numpy(np.float64)
    y = train.material.astype(str).to_numpy()
    groups = train.segmentation_id.to_numpy()
    folds = list(GroupKFold(n_splits=N_FOLDS).split(X, y, groups))
    assert sum(len(set(groups[a]) & set(groups[b])) for a, b in folds) == 0
    checks["n_training"] = int(len(train))
    checks["group_leakage_between_folds"] = 0
    print(f"training {len(train):,} voxels, {len(np.unique(groups))} groups, seed {SEED}")

    # ---- Pill2, loaded only after every model has been fitted --------------
    methods, results, fitted = build_methods(), {}, {}
    for name, factory in methods.items():
        t0 = time.time()
        fold_scores = []
        for k, (tr, te) in enumerate(folds):
            if len(set(y[te])) < 2:
                continue
            m = factory().fit(X[tr], y[tr])
            pred = m.predict(X[te])
            fold_scores.append((balanced_accuracy_score(y[te], pred),
                                f1_score(y[te], pred, average="macro", zero_division=0,
                                         labels=sorted(set(y[te])))))
        final = factory().fit(X, y)
        fitted[name] = final
        results[name] = {"classifier": name, "train_n": int(len(X)),
                         "cv_balanced_accuracy": float(np.mean([s[0] for s in fold_scores])),
                         "cv_macro_f1": float(np.mean([s[1] for s in fold_scores])),
                         "cv_folds_scorable": len(fold_scores), "cv_folds": N_FOLDS,
                         "natural_7_class": bool(final.natural_7_class),
                         "note": final.note, "fit_seconds": round(time.time() - t0, 1)}
        if name == "gmm":
            results[name]["gmm_n_components"] = int(final.n_components_)
            results[name]["gmm_bic"] = final.bic_
            results[name]["gmm_unmapped_classes"] = list(final.unmapped_)
            print(f"  gmm selected {final.n_components_} components by BIC (no labels); "
                  f"classes with no component: {final.unmapped_ or 'none'}")
        print(f"  {name:14s} CV bal acc {results[name]['cv_balanced_accuracy']:.3f}  "
              f"macro F1 {results[name]['cv_macro_f1']:.3f}  "
              f"({len(fold_scores)}/{N_FOLDS} folds, {time.time()-t0:.0f}s)")

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
    print(f"\nPill2 segmented {len(seg):,} | full volume {len(Xv):,}")

    for name, model in fitted.items():
        d = OUT / name
        assert not (d / "metrics.json").is_file(), f"{name} already exists; refusing to overwrite"
        d.mkdir(parents=True, exist_ok=True)
        dump(model, d / "model.joblib")
        classes_ = list(model.classes_)

        proba = model.predict_proba(Xs)
        dev = float(np.abs(proba.sum(1) - 1).max())
        results[name]["max_prob_sum_deviation"] = dev
        assert dev < 1e-6 and proba.min() >= -1e-12, f"{name} produced invalid probabilities"
        pred = np.asarray(classes_)[proba.argmax(1)]
        truth = seg.material.to_numpy()
        results[name].update({
            "pill2_balanced_accuracy": balanced_accuracy_score(truth, pred),
            "pill2_macro_f1": f1_score(truth, pred, average="macro", zero_division=0,
                                       labels=sorted(set(truth))),
            "pill2_accuracy": accuracy_score(truth, pred), "pill2_n": int(len(seg))})
        labels = sorted(set(truth) | set(pred), key=CLASSES.index)
        cm = confusion_matrix(truth, pred, labels=labels)
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(d / "confusion_counts.csv")
        pct = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100
        pd.DataFrame(pct, index=labels, columns=labels).round(2).to_csv(d / "confusion_percent.csv")
        pd.DataFrame(classification_report(truth, pred, labels=labels, output_dict=True,
                                           zero_division=0)).T.round(4).to_csv(
            d / "classification_report.csv")

        out = seg[["z", "y", "x", "material"]].copy()
        out["predicted"] = pred
        for i, c in enumerate(classes_):
            out[f"p_{c}"] = proba[:, i].astype(np.float32)
        out.to_csv(d / "pill2_segmented_predictions.csv.gz", index=False,
                   float_format="%.5g", compression="gzip")

        pv = model.predict_proba(Xv)
        assert float(np.abs(pv.sum(1) - 1).max()) < 1e-6
        pred_vol = np.zeros(valid.shape, np.int8)
        pred_vol[vz, vy, vx] = pv.argmax(1) + 1
        np.save(d / "pill2_fullvolume_prediction.npy", pred_vol)
        fv = pd.DataFrame({"z": vz.astype(np.int16), "y": vy.astype(np.int16),
                           "x": vx.astype(np.int16),
                           "predicted": np.asarray(classes_)[pv.argmax(1)]})
        for i, c in enumerate(classes_):
            fv[f"p_{c}"] = pv[:, i].astype(np.float32)
        fv.to_csv(d / "pill2_fullvolume_predictions.csv.gz", index=False,
                  float_format="%.5g", compression="gzip")
        if name == "ordinal":
            results[name]["ordinal_monotonicity_violation_rate"] = model.violations_
        (d / "metrics.json").write_text(json.dumps(
            {**results[name], "classes": classes_, "features": FEATURES,
             "label_map_fullvolume": {str(i + 1): c for i, c in enumerate(classes_)}},
            indent=2, default=float))
        print(f"  {name:14s} Pill2 bal acc {results[name]['pill2_balanced_accuracy']:.3f}  "
              f"macro F1 {results[name]['pill2_macro_f1']:.3f}  "
              f"acc {results[name]['pill2_accuracy']:.3f}")
        del proba, pv

    checks["all_probabilities_valid"] = bool(
        max(r["max_prob_sum_deviation"] for r in results.values()) < 1e-6)
    checks["methods_not_naturally_7_class"] = {
        n: r["note"] for n, r in results.items() if not r["natural_7_class"]}
    checks["gmm_component_mapping_source"] = "training labels only, after unsupervised fitting"
    (OUT / "critical_checks_extension.json").write_text(json.dumps(checks, indent=2, default=float))
    print(f"\nsaved to {OUT}\nelapsed {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
