#!/usr/bin/env python3
"""Reorganise the finished classifier metrics into Metrics/<metric>/<classifier>/.

Nothing is recalculated, refitted or retrained. Source files are copied verbatim
with shutil.copy2 and the originals stay where they are. The per-metric summary
CSVs are assembled by reading values out of those same files, and the PNGs plot
those values - no metric is computed from predictions here.

The one exception is stated plainly in the output: Probability_distributions has
no precomputed source, so it is described directly from the already-saved
probability columns. That is summary statistics of stored numbers, not a new
model computation.
"""

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = DATA_ROOT / "Classifiers"
SRC = ROOT / "Classifiers"
DST = ROOT / "Metrics"
IOU = DST / "IoU"
ORDER = ["logistic_regression", "random_forest", "gradient_boosting", "rbf_svm", "knn",
         "lda", "qda", "gmm", "hierarchical", "ordinal"]
CLASSES = ["0", "0.05", "0.1", "0.2", "0.25", "0.3", "tissue"]
PRESENT = ["0", "0.1", "0.25", "0.3", "tissue"]      # classes with Pill2 ground truth
copied: list[tuple[str, str]] = []


def copy(src: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / src.name
    shutil.copy2(src, target)
    copied.append((str(src.relative_to(ROOT)), str(target.relative_to(ROOT))))


def heatmap(frame, title, path, fmt="{:.3f}"):
    from echoviewer.diagnostics import _figure, _save
    fig = _figure(figsize=(1.5 * frame.shape[1] + 4, 0.55 * frame.shape[0] + 2.4))
    ax = fig.subplots()
    values = frame.to_numpy(np.float64)
    im = ax.imshow(values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(frame.shape[1])); ax.set_xticklabels(frame.columns, fontsize=9)
    ax.set_yticks(range(frame.shape[0])); ax.set_yticklabels(frame.index, fontsize=9)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8,
                        color="white" if v < 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.02)
    ax.set_title(title, fontsize=12)
    _save(fig, path, [])


def bars(series, title, ylabel, path):
    from echoviewer.diagnostics import _figure, _save
    fig = _figure(figsize=(11, 5.6))
    ax = fig.subplots()
    order = series.sort_values(ascending=False)
    ax.bar(range(len(order)), order.to_numpy(), color="#4a6fe3")
    for i, v in enumerate(order.to_numpy()):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order.index, rotation=35,
                                                         ha="right", fontsize=9)
    ax.set_ylabel(ylabel); ax.grid(alpha=.25, axis="y")
    ax.set_title(title, fontsize=12)
    _save(fig, path, [])


def main() -> int:
    from echoviewer.diagnostics import _figure, _save

    reports, metas = {}, {}
    for name in ORDER:
        reports[name] = pd.read_csv(SRC / name / "classification_report.csv", index_col=0)
        metas[name] = json.loads((SRC / name / "metrics.json").read_text())

    # ---- Precision / Recall / F1 (Dice) -----------------------------------
    for metric, column, folder in (("Precision", "precision", "Precision"),
                                   ("Recall", "recall", "Recall"),
                                   ("F1 (Dice)", "f1-score", "F1_Dice")):
        out = DST / folder
        rows = {}
        for name in ORDER:
            copy(SRC / name / "classification_report.csv", out / name)
            r = reports[name]
            rows[name] = {c: (r.loc[c, column] if c in r.index and r.loc[c, "support"] > 0
                              else np.nan) for c in CLASSES}
        table = pd.DataFrame(rows).T.reindex(ORDER)[CLASSES]
        table.round(6).to_csv(out / f"{folder.lower()}_summary.csv")
        heatmap(table[PRESENT], f"Pill2 per-class {metric} — blank classes have no ground truth",
                out / f"{folder.lower()}_summary.png")
        print(f"  {folder}: {len(ORDER)} classifier folders + summary.csv + .png")

    # ---- Macro-F1 and balanced accuracy -----------------------------------
    for metric, key, folder in (("Macro-F1", "pill2_macro_f1", "Macro_F1"),
                                ("Balanced accuracy", "pill2_balanced_accuracy",
                                 "Balanced_accuracy")):
        out = DST / folder
        rows = []
        for name in ORDER:
            copy(SRC / name / "metrics.json", out / name)
            m = metas[name]
            rows.append({"classifier": name, f"pill2_{folder.lower()}": m[key],
                         "cv_value": m.get("cv_macro_f1" if key.endswith("macro_f1")
                                           else "cv_balanced_accuracy"),
                         "cv_folds_scorable": m.get("cv_folds_scorable"),
                         "train_n": m.get("train_n"), "pill2_n": m.get("pill2_n")})
        table = pd.DataFrame(rows)
        table.round(6).to_csv(out / f"{folder.lower()}_summary.csv", index=False)
        bars(table.set_index("classifier")[f"pill2_{folder.lower()}"],
             f"Pill2 {metric}, all 384,315 labelled voxels", metric,
             out / f"{folder.lower()}_summary.png")
        print(f"  {folder}: {len(ORDER)} classifier folders + summary.csv + .png")

    # ---- Confusion matrices ------------------------------------------------
    out = DST / "Confusion_matrix"
    long_rows = []
    for name in ORDER:
        copy(SRC / name / "confusion_counts.csv", out / name)
        copy(SRC / name / "confusion_percent.csv", out / name)
        counts = pd.read_csv(SRC / name / "confusion_counts.csv", index_col=0)
        for t in counts.index:
            for p in counts.columns:
                long_rows.append({"classifier": name, "true": t, "predicted": p,
                                  "count": int(counts.loc[t, p])})
    pd.DataFrame(long_rows).to_csv(out / "confusion_matrix_summary.csv", index=False)
    fig = _figure(figsize=(21, 9))
    axes = fig.subplots(2, 5)
    for ax, name in zip(axes.ravel(), ORDER):
        pct = pd.read_csv(SRC / name / "confusion_percent.csv", index_col=0)
        im = ax.imshow(pct.to_numpy(), cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(pct.shape[1])); ax.set_xticklabels(pct.columns, rotation=45, fontsize=6)
        ax.set_yticks(range(pct.shape[0])); ax.set_yticklabels(pct.index, fontsize=6)
        for i in range(pct.shape[0]):
            for j in range(pct.shape[1]):
                v = pct.iloc[i, j]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5,
                        color="white" if v > 50 else "black")
        ax.set_title(name, fontsize=9); ax.set_xlabel("predicted", fontsize=7)
        ax.set_ylabel("true", fontsize=7)
    fig.suptitle("Pill2 confusion matrices, row-normalised percent", fontsize=14)
    _save(fig, out / "confusion_matrix_summary.png", [])
    print(f"  Confusion_matrix: {len(ORDER)} classifier folders + summary.csv + .png")

    # ---- Coverage, read out of the existing IoU results --------------------
    out = DST / "Coverage"
    runs = {"all": IOU / "all_probability", "0.5": IOU / "0.5_probability",
            "0.7": IOU / "0.7_probability", "0.75": IOU / "0.75_probability",
            "0.8": IOU / "0.8_probability"}
    runs = {k: v for k, v in runs.items() if (v / "iou_summary.csv").is_file()}
    rows = []
    for threshold, folder in runs.items():
        s = pd.read_csv(folder / "iou_summary.csv", dtype={"class": str})
        macro = s[s["class"] == "macro_mean"]
        for _, r in macro.iterrows():
            rows.append({"classifier": r.classifier, "threshold": threshold,
                         "retained_voxels": int(r.support_voxels), "coverage_pct": r.coverage})
    table = pd.DataFrame(rows)
    wide = table.pivot(index="classifier", columns="threshold",
                       values="coverage_pct").reindex(ORDER)
    wide = wide[[c for c in ["all", "0.5", "0.7", "0.75", "0.8"] if c in wide.columns]]
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "coverage_summary.csv", index=False)
    wide.round(4).to_csv(out / "coverage_by_threshold.csv")
    for name in ORDER:
        sub = table[table.classifier == name]
        (out / name).mkdir(parents=True, exist_ok=True)
        sub.to_csv(out / name / "coverage.csv", index=False)
    fig = _figure(figsize=(11, 6))
    ax = fig.subplots()
    xs = list(range(wide.shape[1]))
    for name in ORDER:
        ax.plot(xs, wide.loc[name].to_numpy(), marker="o", ms=4, lw=1.3, label=name)
    ax.set_xticks(xs); ax.set_xticklabels(wide.columns)
    ax.set_xlabel("confidence threshold"); ax.set_ylabel("coverage (% of labelled Pill2 voxels)")
    ax.grid(alpha=.25); ax.legend(fontsize=7, ncol=2)
    ax.set_title("Coverage retained at each confidence threshold")
    _save(fig, out / "coverage_summary.png", [])
    print(f"  Coverage: {len(ORDER)} classifier folders + summary.csv + .png "
          f"(values read from the existing IoU runs)")

    # ---- Probability distributions ----------------------------------------
    out = DST / "Probability_distributions"
    rows, maxes = [], {}
    for name in ORDER:
        d = pd.read_csv(SRC / name / "pill2_segmented_predictions.csv.gz",
                        dtype={"material": str})
        proba = d[[f"p_{c}" for c in CLASSES]].to_numpy(np.float64)
        best = proba.max(1)
        maxes[name] = best
        per = []
        for i, c in enumerate(CLASSES):
            p = proba[:, i]
            on_true = p[d.material.to_numpy() == c]
            per.append({"classifier": name, "class": c, "n_truth": int(len(on_true)),
                        "mean_p": float(p.mean()), "p05": float(np.percentile(p, 5)),
                        "median_p": float(np.median(p)), "p95": float(np.percentile(p, 95)),
                        "mean_p_on_true_class": float(on_true.mean()) if len(on_true) else np.nan})
        rows.extend(per)
        (out / name).mkdir(parents=True, exist_ok=True)
        pd.DataFrame(per).round(6).to_csv(out / name / "probability_distribution.csv",
                                          index=False)
        del d, proba
    pd.DataFrame(rows).round(6).to_csv(out / "probability_distributions_summary.csv", index=False)
    fig = _figure(figsize=(13, 6.4))
    ax = fig.subplots()
    data = [maxes[n] for n in ORDER]
    parts = ax.violinplot(data, showextrema=False, widths=0.85)
    for body in parts["bodies"]:
        body.set_facecolor("#4a6fe3"); body.set_alpha(.7)
    for pos, v in enumerate(data, start=1):
        q1, q3 = np.percentile(v, [25, 75])
        ax.vlines(pos, q1, q3, color="black", lw=3)
        ax.plot(pos, np.median(v), "o", color="white", markeredgecolor="black", ms=5)
    ax.set_xticks(range(1, len(ORDER) + 1)); ax.set_xticklabels(ORDER, rotation=35,
                                                               ha="right", fontsize=9)
    ax.set_ylabel("max class probability per voxel"); ax.set_ylim(0, 1.02)
    ax.grid(alpha=.25, axis="y")
    ax.set_title("Pill2 predicted-probability distributions, all 384,315 labelled voxels")
    _save(fig, out / "probability_distributions_summary.png", [])
    print(f"  Probability_distributions: {len(ORDER)} classifier folders + summary.csv + .png "
          f"(described from the saved probability columns; no precomputed source existed)")

    manifest = pd.DataFrame(copied, columns=["source", "destination"])
    manifest.to_csv(DST / "copy_manifest.csv", index=False)
    print(f"\n{len(copied)} files copied verbatim; manifest at Metrics/copy_manifest.csv")
    print("no model was fitted, refitted or retrained; no metric value was altered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
