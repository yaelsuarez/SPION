#!/usr/bin/env python3
"""One-vs-rest ROC AUC on Pill2, from saved probabilities only.

Nothing is refitted: every number comes from the per-classifier
`pill2_segmented_predictions.csv.gz` files already on disk. Writes into a new
AUC/ directory and touches nothing that exists.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = DATA_ROOT / "Classifiers"
AUC_DIR = OUT / "AUC"
ORDER = ["logistic_regression", "random_forest", "gradient_boosting", "rbf_svm", "knn",
         "lda", "qda", "gmm", "hierarchical", "ordinal"]
CLASSES = ["0", "0.05", "0.1", "0.2", "0.25", "0.3", "tissue"]


def main() -> int:
    from sklearn.metrics import roc_auc_score, roc_curve
    from echoviewer.diagnostics import LABEL_COLORS, _figure, _save

    AUC_DIR.mkdir(parents=True, exist_ok=True)
    rows, long_rows, curves, notes = [], [], {}, {}

    for name in ORDER:
        path = OUT / name / "pill2_segmented_predictions.csv.gz"
        if not path.is_file():
            continue
        # material must be forced to str: pandas reads "0"/"0.05" as numbers and
        # "tissue" as text, producing a mixed-type target.
        d = pd.read_csv(path, dtype={"material": str})
        truth = d.material.to_numpy()
        cols = [f"p_{c}" for c in CLASSES]
        missing = [c for c in cols if c not in d.columns]
        assert not missing, f"{name} is missing probability columns {missing}"
        proba = d[cols].to_numpy(np.float64)

        # The prediction CSVs were written with %.5g, so a row sums to 1 only to
        # five significant digits. Across seven classes that allows a few 1e-4.
        # The in-memory probabilities were exact; this is storage rounding.
        dev = float(np.abs(proba.sum(1) - 1).max())
        assert dev < 1e-3 and proba.min() >= -1e-12, f"{name} probabilities invalid ({dev})"

        entry = {"classifier": name}
        aucs, supports = {}, {}
        for i, c in enumerate(CLASSES):
            positive = (truth == c)
            n = int(positive.sum())
            supports[c] = n
            if n == 0 or n == len(truth):
                entry[f"auc_class_{c}"] = np.nan   # no positives: AUC undefined
                aucs[c] = np.nan
            else:
                a = float(roc_auc_score(positive, proba[:, i]))
                entry[f"auc_class_{c}"] = a
                aucs[c] = a
                fpr, tpr, _ = roc_curve(positive, proba[:, i])
                step = max(1, len(fpr) // 2000)     # thin for plotting only
                curves[(name, c)] = (fpr[::step], tpr[::step])
            long_rows.append({"classifier": name, "material": c, "support": n,
                              "auc": aucs[c],
                              "prevalence": n / len(truth) if len(truth) else np.nan})

        defined = [c for c in CLASSES if np.isfinite(aucs[c])]
        entry["auc_macro"] = float(np.mean([aucs[c] for c in defined]))
        w = np.array([supports[c] for c in defined], float)
        entry["auc_weighted"] = float(np.average([aucs[c] for c in defined], weights=w))
        entry["n_voxels"] = int(len(truth))
        entry["classes_scored"] = len(defined)
        entry["max_prob_sum_deviation"] = dev
        rows.append(entry)
        notes[name] = {"undefined_classes": [c for c in CLASSES if not np.isfinite(aucs[c])],
                       "supports": supports}
        print(f"  {name:20s} macro {entry['auc_macro']:.4f}  weighted "
              f"{entry['auc_weighted']:.4f}  ({len(defined)} classes scored)")
        del d, proba

    columns = (["classifier"] + [f"auc_class_{c}" for c in CLASSES]
               + ["auc_macro", "auc_weighted", "classes_scored", "n_voxels",
                  "max_prob_sum_deviation"])
    summary = pd.DataFrame(rows)[columns]
    target = AUC_DIR / "auc_summary.csv"
    assert not target.is_file(), "auc_summary.csv already exists; refusing to overwrite"
    summary.round(6).to_csv(target, index=False)
    pd.DataFrame(long_rows).round(6).to_csv(AUC_DIR / "auc_per_class_long.csv", index=False)

    fig = _figure(figsize=(17, 8))
    axes = fig.subplots(2, 3)
    scored = [c for c in CLASSES if any(np.isfinite(r.get(f"auc_class_{c}", np.nan)) for r in rows)]
    for ax, c in zip(axes.ravel(), scored):
        for name in ORDER:
            if (name, c) in curves:
                fpr, tpr = curves[(name, c)]
                ax.plot(fpr, tpr, lw=1.2, label=f"{name} "
                        f"({summary.loc[summary.classifier == name, f'auc_class_{c}'].iloc[0]:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="#999", lw=0.8)
        ax.set_title(f"class {c}", fontsize=11, color=LABEL_COLORS.get(c, "black"))
        ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
        ax.grid(alpha=.25); ax.legend(fontsize=6, loc="lower right")
    for ax in axes.ravel()[len(scored):]:
        ax.set_axis_off()
    fig.suptitle("Pill2 one-vs-rest ROC, all labelled voxels, saved probabilities only",
                 fontsize=14)
    _save(fig, AUC_DIR / "roc_curves.png", [])

    (AUC_DIR / "notes.json").write_text(json.dumps({
        "source": "saved pill2_segmented_predictions.csv.gz; no model was refitted",
        "class_order": CLASSES,
        "undefined_auc_reason": ("0.05 and 0.2 have no ground-truth voxels in Pill2, so "
                                 "one-vs-rest AUC is undefined and left blank; macro and "
                                 "weighted averages are taken over the 5 scored classes"),
        "probability_check": ("every model's probabilities sum to 1 within the precision of "
                              "the stored files, which were written with %.5g; the "
                              "per-classifier deviation is in max_prob_sum_deviation and is "
                              "storage rounding, not a modelling defect. AUC is rank-based "
                              "within a class column, so this rounding is immaterial."),
        "per_classifier": notes,
    }, indent=2, default=float))
    pd.set_option("display.width", 220)
    print("\n" + summary.round(4).to_string(index=False))
    print(f"\nsaved to {AUC_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
