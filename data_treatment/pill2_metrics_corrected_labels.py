#!/usr/bin/env python3
"""Every Pill2 metric, recomputed with Segmentation_4_0.25 relabelled as 0.2.

A parallel tree under Classifiers/Metrics_corrected_labels/. The originals in
Metrics/ are read but never written to.

Only the ground-truth label changes. Predictions and probabilities come from the
saved per-classifier files exactly as stored: no classifier is retrained, no
probability map regenerated, no fit repeated.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data_treatment.segmentation_metrics import (SHAPE, VOXEL_MM3,  # noqa: E402
                                                 distance_pair)

MRI = DATA_ROOT
SRC = MRI / "Classifiers" / "Classifiers"
DST = MRI / "Classifiers" / "Metrics_corrected_labels"
CLASSES = ["0", "0.05", "0.1", "0.2", "0.25", "0.3", "tissue"]
RELABEL = {"0.25": "0.2"}
ORDER = ["logistic_regression", "random_forest", "gradient_boosting", "rbf_svm", "knn",
         "lda", "qda", "gmm", "hierarchical", "ordinal"]
THRESHOLDS = [("all_probability", 0.0), ("0.5_probability", 0.5), ("0.7_probability", 0.7),
              ("0.75_probability", 0.75), ("0.8_probability", 0.8)]


def heat(frame, title, path, cmap="viridis", fmt="{:.3f}", vmin=None, vmax=None):
    from echoviewer.diagnostics import _figure, _save
    fig = _figure(figsize=(1.45 * frame.shape[1] + 5, 0.55 * frame.shape[0] + 2.6))
    ax = fig.subplots()
    values = frame.to_numpy(np.float64)
    finite = values[np.isfinite(values)]
    im = ax.imshow(values, cmap=cmap,
                   vmin=vmin if vmin is not None else (finite.min() if finite.size else 0),
                   vmax=vmax if vmax is not None else (finite.max() if finite.size else 1),
                   aspect="auto")
    ax.set_xticks(range(frame.shape[1])); ax.set_xticklabels(frame.columns, fontsize=9)
    ax.set_yticks(range(frame.shape[0])); ax.set_yticklabels(frame.index, fontsize=9)
    mid = (finite.min() + finite.max()) / 2 if finite.size else 0.5
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            ax.text(j, i, fmt.format(v) if np.isfinite(v) else "–", ha="center", va="center",
                    fontsize=7.5, color="white" if np.isfinite(v) and v < mid else "black")
    fig.colorbar(im, ax=ax, fraction=0.02)
    ax.set_title(title, fontsize=11)
    _save(fig, path, [])


def bars(series, title, ylabel, path):
    from echoviewer.diagnostics import _figure, _save
    fig = _figure(figsize=(11, 5.6))
    ax = fig.subplots()
    order = series.sort_values(ascending=False)
    ax.bar(range(len(order)), order.to_numpy(), color="#4a6fe3")
    for i, v in enumerate(order.to_numpy()):
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order.index, rotation=35,
                                                         ha="right", fontsize=9)
    ax.set_ylabel(ylabel); ax.grid(alpha=.25, axis="y")
    ax.set_title(title, fontsize=12)
    _save(fig, path, [])


def main() -> int:
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 classification_report, confusion_matrix, f1_score,
                                 matthews_corrcoef, roc_auc_score)

    started = time.time()
    assert not DST.exists(), f"{DST} already exists; refusing to overwrite"
    DST.mkdir(parents=True)

    data = {}
    for name in ORDER:
        d = pd.read_csv(SRC / name / "pill2_segmented_predictions.csv.gz",
                        dtype={"material": str, "predicted": str})
        d["material"] = d.material.replace(RELABEL)      # the correction, truth only
        data[name] = d
    n_voxels = len(data[ORDER[0]])
    print(f"{len(data)} classifiers, {n_voxels:,} labelled Pill2 voxels, "
          f"0.25 -> 0.2 applied to the ground truth only\n")

    prec, rec, f1t, summary = {}, {}, {}, []
    auc_rows, mcc_rows, brier_rows, vol_rows, dist_rows = [], [], [], [], []
    iou_tables = {tag: [] for tag, _ in THRESHOLDS}

    for name, d in data.items():
        truth, pred = d.material.to_numpy(), d.predicted.to_numpy()
        proba = d[[f"p_{c}" for c in CLASSES]].to_numpy(np.float64)
        best = proba.max(1)

        labels = sorted(set(truth) | set(pred), key=CLASSES.index)
        rep = classification_report(truth, pred, labels=labels, output_dict=True,
                                    zero_division=0)
        prec[name] = {c: (rep[c]["precision"] if c in rep and rep[c]["support"] else np.nan)
                      for c in CLASSES}
        rec[name] = {c: (rep[c]["recall"] if c in rep and rep[c]["support"] else np.nan)
                     for c in CLASSES}
        f1t[name] = {c: (rep[c]["f1-score"] if c in rep and rep[c]["support"] else np.nan)
                     for c in CLASSES}

        cm = confusion_matrix(truth, pred, labels=labels)
        folder = DST / "Confusion_matrix" / name
        folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cm, index=labels, columns=labels).to_csv(folder / "confusion_counts.csv")
        pct = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100
        pd.DataFrame(pct, index=labels, columns=labels).round(2).to_csv(
            folder / "confusion_percent.csv")
        pd.DataFrame(rep).T.round(4).to_csv(folder / "classification_report.csv")

        # ---- AUC ----------------------------------------------------------
        aucs, supports = {}, {}
        for i, c in enumerate(CLASSES):
            positive = truth == c
            supports[c] = int(positive.sum())
            aucs[c] = (float(roc_auc_score(positive, proba[:, i]))
                       if 0 < positive.sum() < len(truth) else np.nan)
        defined = [c for c in CLASSES if np.isfinite(aucs[c])]
        auc_rows.append({"classifier": name,
                         **{f"auc_class_{c}": aucs[c] for c in CLASSES},
                         "auc_macro": float(np.mean([aucs[c] for c in defined])),
                         "auc_weighted": float(np.average(
                             [aucs[c] for c in defined],
                             weights=[supports[c] for c in defined])),
                         "classes_scored": len(defined), "n_voxels": n_voxels})

        # ---- IoU at every threshold ---------------------------------------
        for tag, threshold in THRESHOLDS:
            keep = best >= threshold if threshold > 0 else np.ones(len(best), bool)
            tk, pk = truth[keep], pred[keep]
            entry = {"classifier": name, "retained_voxels": int(keep.sum()),
                     "coverage_pct": round(100 * keep.sum() / len(best), 4)}
            values = []
            for c in CLASSES:
                t, p = tk == c, pk == c
                union = int((t | p).sum())
                iou = np.nan if union == 0 else int((t & p).sum()) / union
                entry[f"IoU_{c}"] = iou
                if np.isfinite(iou):
                    values.append(iou)
            entry["IoU_macro_mean"] = float(np.mean(values)) if values else np.nan
            entry["classes_scored"] = len(values)
            iou_tables[tag].append(entry)

        # ---- MCC, Brier, volume error, distances --------------------------
        onehot = np.zeros_like(proba)
        for i, c in enumerate(CLASSES):
            onehot[:, i] = truth == c
        mcc_all = float(matthews_corrcoef(truth, pred))
        multiclass_brier = float(((proba - onehot) ** 2).sum(1).mean())

        per_mcc, per_brier, hd95s, assds = {}, {}, {}, {}
        z, y, x = d.z.to_numpy(np.intp), d.y.to_numpy(np.intp), d.x.to_numpy(np.intp)
        for i, c in enumerate(CLASSES):
            t, p = truth == c, pred == c
            n_true, n_pred = int(t.sum()), int(p.sum())

            diff = n_pred - n_true
            vol_rows.append({"classifier": name, "class": c, "n_true_voxels": n_true,
                             "n_predicted_voxels": n_pred,
                             "true_volume_mm3": n_true * VOXEL_MM3,
                             "predicted_volume_mm3": n_pred * VOXEL_MM3,
                             "signed_error_voxels": diff,
                             "absolute_error_voxels": abs(diff),
                             "absolute_error_mm3": abs(diff) * VOXEL_MM3,
                             "relative_error": (diff / n_true) if n_true else np.nan,
                             "relative_error_pct": (100 * diff / n_true) if n_true else np.nan,
                             "status": "ok" if n_true else "no ground-truth voxels"})

            b = float(((proba[:, i] - t) ** 2).mean())
            per_brier[c] = b
            brier_rows.append({"classifier": name, "class": c, "brier": b,
                               "n_true_voxels": n_true,
                               "mean_probability": float(proba[:, i].mean()),
                               "prevalence": n_true / n_voxels,
                               "status": "ok" if n_true else
                                         "no ground-truth voxels: false confidence only"})

            if n_true == 0 and n_pred == 0:
                m, status = np.nan, "class absent from truth and prediction: undefined"
            elif n_true in (0, n_voxels) or n_pred in (0, n_voxels):
                m, status = 0.0, "one side is constant: MCC is exactly 0 by definition"
            else:
                m, status = float(matthews_corrcoef(t, p)), "ok"
            per_mcc[c] = m
            mcc_rows.append({"classifier": name, "class": c, "mcc_ovr": m,
                             "n_true_voxels": n_true, "n_predicted_voxels": n_pred,
                             "status": status})

            if n_true == 0 or n_pred == 0:
                hd = assd = np.nan
                status = "no ground-truth voxels" if n_true == 0 else "class never predicted"
            else:
                tv = np.zeros(SHAPE, bool); tv[z[t], y[t], x[t]] = True
                pv = np.zeros(SHAPE, bool); pv[z[p], y[p], x[p]] = True
                dist = distance_pair(tv, pv)
                if dist is None:
                    hd = assd = np.nan; status = "no extractable surface"
                else:
                    hd, assd, status = float(np.percentile(dist, 95)), float(dist.mean()), "ok"
                del tv, pv
            hd95s[c], assds[c] = hd, assd
            dist_rows.append({"classifier": name, "class": c, "hd95_mm": hd, "assd_mm": assd,
                              "n_true_voxels": n_true, "n_predicted_voxels": n_pred,
                              "spacing_mm": "0.32 x 0.32 x 0.32", "status": status})

        ok = lambda dd: [v for v in dd.values() if np.isfinite(v)]
        vol = pd.DataFrame([r for r in vol_rows if r["classifier"] == name])
        summary.append({
            "classifier": name,
            "balanced_accuracy": balanced_accuracy_score(truth, pred),
            "macro_f1": f1_score(truth, pred, average="macro", zero_division=0,
                                 labels=sorted(set(truth))),
            "accuracy": accuracy_score(truth, pred),
            "auc_macro": auc_rows[-1]["auc_macro"], "auc_weighted": auc_rows[-1]["auc_weighted"],
            "iou_macro_all_voxels": iou_tables["all_probability"][-1]["IoU_macro_mean"],
            "mcc_multiclass": mcc_all, "mcc_ovr_macro": float(np.mean(ok(per_mcc))),
            "brier_multiclass": multiclass_brier,
            "brier_macro": float(np.mean(list(per_brier.values()))),
            "hd95_macro_mm": float(np.mean(ok(hd95s))) if ok(hd95s) else np.nan,
            "assd_macro_mm": float(np.mean(ok(assds))) if ok(assds) else np.nan,
            "mean_abs_relative_volume_error": float(np.nanmean(np.abs(vol.relative_error))),
            "n_voxels": n_voxels,
        })
        print(f"  {name:20s} bal acc {summary[-1]['balanced_accuracy']:.3f}  "
              f"macro F1 {summary[-1]['macro_f1']:.3f}  IoU {summary[-1]['iou_macro_all_voxels']:.3f}  "
              f"MCC {mcc_all:.3f}  ({time.time()-started:.0f}s)")
        del proba, onehot

    # ---- write everything --------------------------------------------------
    def table(d, folder, stem, title, **kw):
        out = DST / folder
        out.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(d).T.reindex(ORDER)[CLASSES]
        frame.round(6).to_csv(out / f"{stem}.csv")
        present = [c for c in CLASSES if frame[c].notna().any()]
        heat(frame[present], title, out / f"{stem}.png", **kw)
        return frame

    table(prec, "Precision", "precision_summary", "Pill2 per-class precision, corrected labels",
          vmin=0, vmax=1)
    table(rec, "Recall", "recall_summary", "Pill2 per-class recall, corrected labels",
          vmin=0, vmax=1)
    table(f1t, "F1_Dice", "f1_dice_summary", "Pill2 per-class F1 (Dice), corrected labels",
          vmin=0, vmax=1)

    frame = pd.DataFrame(summary)
    frame.round(6).to_csv(DST / "summary.csv", index=False)
    for folder, column, label in (("Balanced_accuracy", "balanced_accuracy", "Balanced accuracy"),
                                  ("Macro_F1", "macro_f1", "Macro-F1")):
        (DST / folder).mkdir(parents=True, exist_ok=True)
        frame[["classifier", column]].round(6).to_csv(
            DST / folder / f"{folder.lower()}_summary.csv", index=False)
        bars(frame.set_index("classifier")[column],
             f"Pill2 {label}, corrected labels", label,
             DST / folder / f"{folder.lower()}_summary.png")

    (DST / "AUC").mkdir(parents=True, exist_ok=True)
    auc = pd.DataFrame(auc_rows)
    auc.round(6).to_csv(DST / "AUC" / "auc_summary.csv", index=False)
    heat(auc.set_index("classifier")[[f"auc_class_{c}" for c in CLASSES]].rename(
        columns=lambda s: s.replace("auc_class_", "")).dropna(axis=1, how="all"),
        "Pill2 one-vs-rest AUC, corrected labels", DST / "AUC" / "auc_summary.png",
        vmin=0.5, vmax=1)

    for tag, _ in THRESHOLDS:
        out = DST / "IoU" / tag
        out.mkdir(parents=True, exist_ok=True)
        t = pd.DataFrame(iou_tables[tag]).set_index("classifier").reindex(ORDER)
        t.round(6).to_csv(out / "iou_summary.csv")
        heat(t[[f"IoU_{c}" for c in CLASSES]].rename(columns=lambda s: s.replace("IoU_", "")),
             f"Pill2 per-class IoU ({tag.replace('_', ' ')}), corrected labels",
             out / "iou_by_class.png", vmin=0, vmax=1)

    for rows_, folder, stem, column, title, kw in (
            (mcc_rows, "MCC", "mcc", "mcc_ovr", "Pill2 one-vs-rest MCC, corrected labels",
             dict(vmin=0, vmax=1)),
            (brier_rows, "Brier", "brier", "brier", "Pill2 Brier per class, corrected labels",
             dict(cmap="viridis_r")),
            (vol_rows, "Volume_error", "volume_error", "relative_error_pct",
             "Pill2 volume error (%), corrected labels",
             dict(cmap="coolwarm", fmt="{:+.0f}", vmin=-200, vmax=200)),
            (dist_rows, "HD95", "hd95", "hd95_mm", "Pill2 HD95 (mm), corrected labels",
             dict(cmap="magma_r", fmt="{:.1f}")),
            (dist_rows, "ASSD", "assd", "assd_mm", "Pill2 ASSD (mm), corrected labels",
             dict(cmap="magma_r", fmt="{:.2f}"))):
        out = DST / folder
        out.mkdir(parents=True, exist_ok=True)
        long = pd.DataFrame(rows_)
        long.round(6).to_csv(out / f"{stem}.csv", index=False)
        wide = long.pivot(index="classifier", columns="class", values=column).reindex(ORDER)
        heat(wide[[c for c in CLASSES if c in wide.columns]], title,
             out / f"{stem}.png", **kw)

    (DST / "README.json").write_text(json.dumps({
        "correction": "Pill2 ground-truth label 0.25 treated as 0.2",
        "affected_segmentation": "Segmentations/Pill2/segmentations/Segmentation_4_0.25",
        "scope": "ground truth only; predictions and probabilities unchanged",
        "retrained": False, "probabilities_regenerated": False, "refitted": False,
        "source": "Classifiers/Classifiers/<name>/pill2_segmented_predictions.csv.gz",
        "originals": "Classifiers/Metrics/ holds the uncorrected results, untouched",
        "class_order": CLASSES, "n_voxels": n_voxels,
        "voxel_spacing_mm": [0.32, 0.32, 0.32],
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2, default=float))

    pd.set_option("display.width", 250)
    print("\n" + frame.round(4).to_string(index=False))
    print(f"\nsaved to {DST}\nelapsed {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
