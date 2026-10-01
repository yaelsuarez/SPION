#!/usr/bin/env python3
"""Rebuild Classifiers/summary.csv from the saved per-classifier artifacts.

Pill2 contains no `0.2` ground truth, so a "0.2 recall" column reports a
zero-support row and means nothing. The observable half of that confusion is
0.25 predicted as 0.2, which is what this records. Nothing is refitted.
"""

import json
from pathlib import Path

import pandas as pd
from config import DATA_ROOT  # noqa: E402

OUT = DATA_ROOT / "Classifiers"
ORDER = ["logistic_regression", "random_forest", "gradient_boosting", "rbf_svm", "knn",
         "lda", "qda", "gmm", "hierarchical", "ordinal"]
PREVIOUS = OUT / "summary_first_five_classifiers.csv"


def main() -> int:
    # Keep the five-classifier table that existed before the extension.
    if (OUT / "summary.csv").is_file() and not PREVIOUS.is_file():
        PREVIOUS.write_bytes((OUT / "summary.csv").read_bytes())
        print(f"preserved the previous table as {PREVIOUS.name}")

    rows, per_class = [], []
    for name in ORDER:
        d = OUT / name
        m = json.loads((d / "metrics.json").read_text())
        pct = pd.read_csv(d / "confusion_percent.csv", index_col=0)
        rep = pd.read_csv(d / "classification_report.csv", index_col=0)
        cell = lambda a, b: (float(pct.loc[a, b])
                             if a in pct.index and b in pct.columns else float("nan"))
        recall = lambda c: (float(rep.loc[c, "recall"])
                            if c in rep.index and rep.loc[c, "support"] > 0 else float("nan"))
        rows.append({
            "classifier": name, "train_n": m["train_n"],
            "cv_balanced_accuracy": m["cv_balanced_accuracy"], "cv_macro_f1": m["cv_macro_f1"],
            "cv_folds_scorable": m["cv_folds_scorable"],
            "pill2_balanced_accuracy": m["pill2_balanced_accuracy"],
            "pill2_macro_f1": m["pill2_macro_f1"],
            "pill2_accuracy": m.get("pill2_accuracy", float("nan")),
            "pill2_n": m["pill2_n"],
            "natural_7_class": m.get("natural_7_class", True),
            "recall_0": recall("0"), "recall_0.1": recall("0.1"),
            "recall_0.25": recall("0.25"), "recall_0.3": recall("0.3"),
            "recall_tissue": recall("tissue"),
            "pct_0.25_predicted_0.2": cell("0.25", "0.2"),
            "pct_0.25_correct": cell("0.25", "0.25"),
            "pct_tissue_predicted_0.1": cell("tissue", "0.1"),
            "pct_tissue_predicted_0.05": cell("tissue", "0.05"),
            "pct_tissue_correct": cell("tissue", "tissue"),
            "max_prob_sum_deviation": m["max_prob_sum_deviation"],
            "fit_seconds": m["fit_seconds"],
            "note": m.get("note", ""),
        })
        for cls in rep.index:
            if cls in ("accuracy", "macro avg", "weighted avg") or rep.loc[cls, "support"] == 0:
                continue
            per_class.append({"classifier": name, "material": cls,
                              "precision": rep.loc[cls, "precision"],
                              "recall": rep.loc[cls, "recall"],
                              "f1": rep.loc[cls, "f1-score"],
                              "support": int(rep.loc[cls, "support"])})

    sm = pd.DataFrame(rows)
    best = sm.sort_values(["cv_balanced_accuracy", "cv_macro_f1"], ascending=False).classifier.iloc[0]
    sm["selected_by_cv"] = sm.classifier == best
    sm["selection_metric"] = "cv_balanced_accuracy (chosen before any Pill2 result)"
    sm.round(4).to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(per_class).round(4).to_csv(OUT / "per_class_metrics_all_classifiers.csv",
                                            index=False)
    pd.set_option("display.width", 250)
    print(sm.round(3).to_string(index=False))
    print(f"\nselected by CV: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
