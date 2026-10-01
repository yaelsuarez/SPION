#!/usr/bin/env python3
"""Rank the top three classifiers for every metric already computed under Metrics/.

Reads existing metric CSVs only. Nothing is retrained, refitted, recalculated or
modified. The single derived quantity is the confusion matrix's scalar summary,
overall accuracy, which is the trace of the stored count matrix over its total -
a sum of numbers already on disk, not a re-evaluation of any prediction.

Each cell carries the value alongside the model name so a row can be read
without opening the source file. Rows where every model ties - the classes with
no Pill2 ground truth - are labelled rather than given a spurious winner.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = DATA_ROOT / "Classifiers"
M = ROOT / "Metrics"
TARGET = ROOT / "metric_top3_models.csv"
ORDER = ["logistic_regression", "random_forest", "gradient_boosting", "rbf_svm", "knn",
         "lda", "qda", "gmm", "hierarchical", "ordinal"]
KEY_CLASSES = ["0", "0.1", "0.2", "0.25", "0.3", "tissue"]

rows: list[dict] = []


def add(parameter, series, direction, fmt="{:.4f}"):
    """Rank one metric. direction: 'max' or 'min'. Values already computed."""
    s = pd.Series(series).astype(float)
    s = s[np.isfinite(s)]
    entry = {"Parameter": parameter, "Best model 1": "", "Best model 2": "",
             "Best model 3": ""}
    if s.empty:
        entry["Best model 1"] = "not rankable: no values available"
        rows.append(entry)
        return
    if s.nunique() == 1:
        entry["Best model 1"] = f"all {len(s)} models tie ({fmt.format(s.iloc[0])})"
        rows.append(entry)
        return
    ranked = s.sort_values(ascending=(direction == "min"))
    for i in range(min(3, len(ranked))):
        entry[f"Best model {i+1}"] = f"{ranked.index[i]} ({fmt.format(ranked.iloc[i])})"
    rows.append(entry)


def main() -> int:
    assert not TARGET.exists(), f"{TARGET.name} already exists; refusing to overwrite"

    # ---- balanced accuracy, macro-F1 --------------------------------------
    ba = pd.read_csv(M / "Balanced_accuracy" / "balanced_accuracy_summary.csv")
    add("Balanced_accuracy_overall",
        ba.set_index("classifier")["pill2_balanced_accuracy"], "max")
    mf = pd.read_csv(M / "Macro_F1" / "macro_f1_summary.csv")
    add("Macro_F1_overall", mf.set_index("classifier")["pill2_macro_f1"], "max")

    # ---- precision / recall / F1 per class --------------------------------
    for label, filename, folder in (("Precision", "precision_summary.csv", "Precision"),
                                    ("Recall", "recall_summary.csv", "Recall"),
                                    ("F1", "f1_dice_summary.csv", "F1_Dice")):
        table = pd.read_csv(M / folder / filename, index_col=0)
        table.columns = [str(c) for c in table.columns]
        add(f"{label}_macro_over_present_classes",
            table[[c for c in table.columns if table[c].notna().any()]].mean(axis=1), "max")
        for c in KEY_CLASSES:
            if c in table.columns:
                add(f"{label}_{c}", table[c], "max")

    # ---- confusion matrix: overall accuracy from the stored counts ---------
    acc = {}
    for name in ORDER:
        counts = pd.read_csv(M / "Confusion_matrix" / name / "confusion_counts.csv", index_col=0)
        counts.index = [str(i) for i in counts.index]
        counts.columns = [str(c) for c in counts.columns]
        total = counts.to_numpy().sum()
        correct = sum(counts.loc[c, c] for c in counts.index if c in counts.columns)
        acc[name] = correct / total
    add("Confusion_matrix_overall_accuracy", acc, "max")

    # ---- AUC ---------------------------------------------------------------
    auc = pd.read_csv(M / "AUC" / "auc_summary.csv").set_index("classifier")
    add("AUC_macro", auc["auc_macro"], "max")
    add("AUC_weighted", auc["auc_weighted"], "max")
    for c in KEY_CLASSES:
        col = f"auc_class_{c}"
        if col in auc.columns:
            add(f"AUC_{c}", auc[col], "max")

    # ---- IoU ---------------------------------------------------------------
    for tag, folder in (("all", "all_probability"), ("0.5", "0.5_probability"),
                        ("0.7", "0.7_probability"), ("0.75", "0.75_probability"),
                        ("0.8", "0.8_probability")):
        path = M / "IoU" / folder / "iou_summary.csv"
        if not path.is_file():
            continue
        iou = pd.read_csv(path, dtype={"class": str})
        macro = iou[iou["class"] == "macro_mean"].set_index("classifier")["IoU"]
        add(f"IoU_macro_at_{tag}", macro, "max")
        if tag == "all":
            for c in KEY_CLASSES:
                sub = iou[iou["class"] == c].set_index("classifier")["IoU"]
                if len(sub):
                    add(f"IoU_{c}", sub, "max")

    # ---- coverage (higher = more of the volume classified) -----------------
    cov = pd.read_csv(M / "Coverage" / "coverage_by_threshold.csv", index_col=0)
    for column in cov.columns:
        add(f"Coverage_at_{column}", cov[column], "max", fmt="{:.2f}")

    # ---- MCC ---------------------------------------------------------------
    summary = pd.read_csv(M / "summary.csv").set_index("classifier")
    add("MCC_multiclass", summary["mcc_multiclass"], "max")
    add("MCC_ovr_macro", summary["mcc_ovr_macro"], "max")
    mcc = pd.read_csv(M / "MCC" / "mcc.csv", dtype={"class": str})
    for c in KEY_CLASSES:
        sub = mcc[mcc["class"] == c].set_index("classifier")["mcc_ovr"]
        if len(sub):
            add(f"MCC_{c}", sub, "max")

    # ---- Brier (lower is better) ------------------------------------------
    add("Brier_multiclass", summary["brier_multiclass"], "min")
    add("Brier_macro", summary["brier_macro"], "min")
    brier = pd.read_csv(M / "Brier" / "brier.csv", dtype={"class": str})
    for c in KEY_CLASSES:
        sub = brier[brier["class"] == c].set_index("classifier")["brier"]
        if len(sub):
            add(f"Brier_{c}", sub, "min")

    # ---- volume error (lower absolute relative error is better) -----------
    add("Volume_error_mean_abs_relative", summary["mean_abs_relative_volume_error"], "min")
    add("Volume_error_total_abs_mm3", summary["total_abs_volume_error_mm3"], "min",
        fmt="{:.0f}")
    vol = pd.read_csv(M / "Volume_error" / "volume_error.csv", dtype={"class": str})
    for c in KEY_CLASSES:
        sub = vol[vol["class"] == c].set_index("classifier")["relative_error_pct"].abs()
        if len(sub):
            add(f"Volume_error_{c}_abs_relative_pct", sub, "min", fmt="{:.1f}")

    # ---- HD95 and ASSD (lower is better) ----------------------------------
    add("HD95_macro_mm", summary["hd95_macro_mm"], "min", fmt="{:.2f}")
    add("ASSD_macro_mm", summary["assd_macro_mm"], "min", fmt="{:.2f}")
    for label, filename, column in (("HD95", "hd95.csv", "hd95_mm"),
                                    ("ASSD", "assd.csv", "assd_mm")):
        table = pd.read_csv(M / label / filename, dtype={"class": str})
        for c in KEY_CLASSES:
            sub = table[table["class"] == c].set_index("classifier")[column]
            if len(sub):
                add(f"{label}_{c}_mm", sub, "min", fmt="{:.2f}")

    out = pd.DataFrame(rows)[["Parameter", "Best model 1", "Best model 2", "Best model 3"]]
    out.to_csv(TARGET, index=False)
    print(out.to_string(index=False))
    print(f"\n{len(out)} rows -> {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
