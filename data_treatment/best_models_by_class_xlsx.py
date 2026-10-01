#!/usr/bin/env python3
"""Per-class top-3 classifier ranking as an Excel workbook, one sheet per class.

Every value is transcribed from a metric CSV already under Metrics/. Nothing is
recalculated, retrained, refitted or modified, so the workbook holds no formulas
- there is no input-to-output relationship inside the sheet to preserve. The
source file for each metric is named at the foot of every sheet.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = DATA_ROOT / "Classifiers"
M = ROOT / "Metrics"
TARGET = ROOT / "best_models_by_class.xlsx"
SHEETS = ["0", "0.1", "0.25", "0.3", "tissue"]
FONT = "Arial"


def load():
    """Every per-class metric, as {metric_label: (frame indexed by classifier, direction, fmt)}."""
    tables = {}

    for label, folder, filename in (("Precision", "Precision", "precision_summary.csv"),
                                    ("Recall", "Recall", "recall_summary.csv"),
                                    ("F1 (Dice)", "F1_Dice", "f1_dice_summary.csv")):
        t = pd.read_csv(M / folder / filename, index_col=0)
        t.columns = [str(c) for c in t.columns]
        tables[label] = (t, "max", "{:.4f}", f"Metrics/{folder}/{filename}")

    auc = pd.read_csv(M / "AUC" / "auc_summary.csv").set_index("classifier")
    auc_wide = auc[[c for c in auc.columns if c.startswith("auc_class_")]]
    auc_wide.columns = [c.replace("auc_class_", "") for c in auc_wide.columns]
    tables["AUC (one-vs-rest)"] = (auc_wide, "max", "{:.4f}", "Metrics/AUC/auc_summary.csv")

    for tag, folder in (("no threshold", "all_probability"), ("p>=0.5", "0.5_probability"),
                        ("p>=0.7", "0.7_probability"), ("p>=0.75", "0.75_probability"),
                        ("p>=0.8", "0.8_probability")):
        path = M / "IoU" / folder / "iou_summary.csv"
        if not path.is_file():
            continue
        iou = pd.read_csv(path, dtype={"class": str})
        iou = iou[iou["class"] != "macro_mean"]
        tables[f"IoU ({tag})"] = (iou.pivot(index="classifier", columns="class", values="IoU"),
                                  "max", "{:.4f}", f"Metrics/IoU/{folder}/iou_summary.csv")

    mcc = pd.read_csv(M / "MCC" / "mcc.csv", dtype={"class": str})
    tables["MCC (one-vs-rest)"] = (mcc.pivot(index="classifier", columns="class",
                                             values="mcc_ovr"),
                                   "max", "{:.4f}", "Metrics/MCC/mcc.csv")

    prob = pd.read_csv(M / "Probability_distributions" /
                       "probability_distributions_summary.csv", dtype={"class": str})
    tables["Mean probability on true class"] = (
        prob.pivot(index="classifier", columns="class", values="mean_p_on_true_class"),
        "max", "{:.4f}", "Metrics/Probability_distributions/probability_distributions_summary.csv")

    brier = pd.read_csv(M / "Brier" / "brier.csv", dtype={"class": str})
    tables["Brier score"] = (brier.pivot(index="classifier", columns="class", values="brier"),
                             "min", "{:.4f}", "Metrics/Brier/brier.csv")

    vol = pd.read_csv(M / "Volume_error" / "volume_error.csv", dtype={"class": str})
    abs_rel = vol.copy()
    abs_rel["v"] = abs_rel.relative_error_pct.abs()
    tables["Volume error, absolute relative (%)"] = (
        abs_rel.pivot(index="classifier", columns="class", values="v"),
        "min", "{:.1f}", "Metrics/Volume_error/volume_error.csv")
    tables["Volume error, absolute (mm3)"] = (
        vol.pivot(index="classifier", columns="class", values="absolute_error_mm3"),
        "min", "{:.1f}", "Metrics/Volume_error/volume_error.csv")

    hd = pd.read_csv(M / "HD95" / "hd95.csv", dtype={"class": str})
    tables["HD95 (mm)"] = (hd.pivot(index="classifier", columns="class", values="hd95_mm"),
                           "min", "{:.2f}", "Metrics/HD95/hd95.csv")
    assd = pd.read_csv(M / "ASSD" / "assd.csv", dtype={"class": str})
    tables["ASSD (mm)"] = (assd.pivot(index="classifier", columns="class", values="assd_mm"),
                           "min", "{:.2f}", "Metrics/ASSD/assd.csv")
    return tables


def main() -> int:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    assert not TARGET.exists(), f"{TARGET.name} already exists; refusing to overwrite"
    tables = load()

    wb = Workbook()
    wb.remove(wb.active)
    header_font = Font(name=FONT, bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="2F5597")
    body = Font(name=FONT)
    note = Font(name=FONT, italic=True, size=9)

    for cls in SHEETS:
        ws = wb.create_sheet(cls)
        ws.append(["Metric", "Best model", "2nd best model", "3rd best model"])
        sources = []
        for label, (frame, direction, fmt, source) in tables.items():
            if cls not in frame.columns:
                continue
            s = pd.to_numeric(frame[cls], errors="coerce").dropna()
            arrow = "higher is better" if direction == "max" else "lower is better"
            name = f"{label} — {arrow}"
            if s.empty:
                ws.append([name, "not rankable: no values available", "", ""])
            elif s.nunique() == 1:
                ws.append([name, f"all {len(s)} models tie ({fmt.format(s.iloc[0])})", "", ""])
            else:
                r = s.sort_values(ascending=(direction == "min"))
                cells = [f"{r.index[i]} ({fmt.format(r.iloc[i])})" if i < len(r) else ""
                         for i in range(3)]
                ws.append([name] + cells)
            sources.append(f"{label}: {source}")

        for cell in ws[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center")
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
            for cell in row:
                cell.font = body
                cell.alignment = Alignment(vertical="center")
        ws.freeze_panes = "A2"
        for i, width in enumerate([44, 34, 34, 34], start=1):
            ws.column_dimensions[get_column_letter(i)].width = width

        foot = ws.max_row + 2
        ws.cell(foot, 1, f"Class '{cls}'. Values transcribed exactly from the metric files "
                         f"below; nothing was recalculated, retrained or refitted.").font = note
        ws.cell(foot + 1, 1, "Ranking direction is stated in each metric name. Distance "
                             "metrics use 0.32 x 0.32 x 0.32 mm isotropic spacing.").font = note
        for j, line in enumerate(sources, start=foot + 3):
            ws.cell(j, 1, f"Source — {line}").font = note

    wb.save(TARGET)
    print(f"saved {TARGET}")
    for cls in SHEETS:
        n = sum(1 for label, (f, *_ ) in tables.items() if cls in f.columns)
        print(f"  sheet '{cls}': {n} metrics ranked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
