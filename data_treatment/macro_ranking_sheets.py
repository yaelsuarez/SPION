#!/usr/bin/env python3
"""Rank the classifiers on the macro averages, with and without tissue.

Adds three sheets to Metrics_corrected_labels/summary.xlsx and touches nothing
else. Ranks are derived from sheets already in that workbook; no metric is
recomputed and no model is refitted.

Excluding tissue is not a cosmetic change: it drops the class that holds 90.3%
of the voxels and that every model handles worst, so the ranking it produces
answers a different question - how well concentrations are separated, given the
voxel is SPION.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from data_treatment.rank_corrected_labels import M  # noqa: E402

TARGET = M / "summary.xlsx"
WITH = "Macro average"
WITHOUT = "Macro average (no tissue)"
#: metric -> True when higher is better
DIRECTION = {"Precision": True, "Recall": True, "F1 / Dice": True, "AUC": True,
             "IoU": True, "Brier score": False, "Voxel volume error (%)": False,
             "HD95 (mm)": False, "ASSD (mm)": False}
FONT = "Arial"


def ranks_of(frame):
    r = pd.DataFrame({m: frame[m].rank(ascending=not up).astype(int)
                      for m, up in DIRECTION.items()})
    r["mean_rank"] = r[list(DIRECTION)].mean(axis=1).round(2)
    r["overall_rank"] = r.mean_rank.rank(method="min").astype(int)
    return r


def main() -> int:
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    started = time.time()
    book = pd.ExcelFile(TARGET)
    a = pd.read_excel(book, WITH).set_index("classifier")
    b = pd.read_excel(book, WITHOUT).set_index("classifier")
    ra, rb = ranks_of(a), ranks_of(b)

    comparison = pd.DataFrame({
        "mean_rank_with_tissue": ra.mean_rank,
        "overall_rank_with_tissue": ra.overall_rank,
        "mean_rank_no_tissue": rb.mean_rank,
        "overall_rank_no_tissue": rb.overall_rank,
    })
    comparison["rank_change"] = (comparison.overall_rank_with_tissue
                                 - comparison.overall_rank_no_tissue)
    comparison = comparison.sort_values("overall_rank_with_tissue")

    wb = load_workbook(TARGET)
    header_font = Font(name=FONT, bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="2F5597")
    for name, frame in (("Ranking macro (with tissue)", ra.loc[comparison.index]),
                        ("Ranking macro (no tissue)", rb.loc[comparison.index]),
                        ("Ranking comparison", comparison)):
        assert name not in wb.sheetnames, f"{name} already exists"
        ws = wb.create_sheet(name)
        ws.append(["classifier"] + [str(c) for c in frame.columns])
        for key, row in frame.iterrows():
            ws.append([key] + [float(v) if isinstance(v, float) else int(v) for v in row])
        for cell in ws[1]:
            cell.font = header_font; cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.font = Font(name=FONT)
        ws.freeze_panes = "B2"
        ws.column_dimensions["A"].width = 24
        for i in range(2, frame.shape[1] + 2):
            ws.column_dimensions[get_column_letter(i)].width = 16

    readme = wb["_README"]
    readme.append(["Rankings", "three sheets rank the classifiers on the macro averages, "
                               "1 = best per metric; mean_rank averages the 9 metric ranks. "
                               "rank_change > 0 means the model improves when tissue is "
                               "excluded"])
    for cell in readme[readme.max_row]:
        cell.font = Font(name=FONT)
        cell.alignment = Alignment(vertical="center", wrap_text=True)
    wb.save(TARGET)

    for name, r in (("WITH TISSUE", ra), ("WITHOUT TISSUE", rb)):
        assert all(sorted(r[m]) == list(range(1, 11)) for m in DIRECTION), name
    pd.set_option("display.width", 230)
    print("RANKING ON MACRO AVERAGES (1 = best)\n")
    print(comparison.to_string())
    print(f"\nsaved to {TARGET}\nelapsed {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
