#!/usr/bin/env python3
"""Rebuild the four label-derived sheets from the current Labels sheet.

Regenerates, in tissue_anchored_segmentation_means.xlsx:
    means_new_labels, weighted_means_by_label, rs_by_organ, i0_by_organ

Those four are replaced because they are derived entirely from Labels and go
stale when it changes. Every other sheet - segmentation_means, anchors, Labels,
_README - is left exactly as it is, and no voxel data is recomputed: all
statistics come from tissue_anchored_voxels_all.csv.gz.

Species is parsed from the END of each label, not the start, so a label like
`ileumpoopmouse` resolves to organ `ileumpoop` rather than being folded into
`ileum`. SDs for combined groups are pooled over the voxel population, not
averaged across segmentations.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HUMAN = DATA_ROOT / "human"
MEANS = HUMAN / "tissue_anchored_segmentation_means.xlsx"
VOXELS = HUMAN / "tissue_anchored_voxels_all.csv.gz"
REBUILD = ["means_new_labels", "weighted_means_by_label", "rs_by_organ", "i0_by_organ"]
KEEP = ["segmentation_means", "anchors", "_README", "Labels"]
ORGANS = ["liver", "stomach", "duodenum", "ileum", "jejunum", "colon"]   # requested order
SPECIES = ["human", "mouse", "pig"]
#: longest first, so "human adult" is matched before "human"
SUFFIXES = ["human adult", "human pediatric", "mouse", "pig"]
FONT = "Arial"


def split_label(label):
    for suffix in SUFFIXES:
        if label.endswith(suffix):
            organ = label[: -len(suffix)].strip()
            species = "human" if suffix.startswith("human") else suffix
            return organ, species
    raise ValueError(f"cannot parse species from {label!r}")


def style(ws, widths):
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    header_font = Font(name=FONT, bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="2F5597")
    for cell in ws[1]:
        cell.font = header_font; cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.font = Font(name=FONT)
    ws.freeze_panes = "B2"
    for i, width in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = width


def write(wb, name, frame, widths):
    ws = wb.create_sheet(name)
    ws.append([str(c) for c in frame.columns])
    for record in frame.itertuples(index=False, name=None):
        ws.append([None if (isinstance(v, float) and np.isnan(v))
                   else (bool(v) if isinstance(v, (bool, np.bool_))
                         else (float(v) if isinstance(v, (int, float, np.number)) else str(v)))
                   for v in record])
    style(ws, widths)


def main() -> int:
    from openpyxl import load_workbook

    started = time.time()
    labels = pd.read_excel(MEANS, "Labels")
    mapping = dict(zip(labels["Old labels"].astype(str), labels["New labels"].astype(str)))

    voxels = pd.read_csv(VOXELS, usecols=["sample_segmentation", "rs_selected",
                                          "i0_tissue_anchored"])
    voxels["new_label"] = voxels.sample_segmentation.map(mapping)
    unmatched = sorted(voxels.loc[voxels.new_label.isna(), "sample_segmentation"].unique())
    voxels["label_or_id"] = voxels.new_label.fillna(voxels.sample_segmentation)
    print(f"{len(voxels):,} voxels, {voxels.sample_segmentation.nunique()} segmentations")
    print(f"unmatched, kept under their own id: {unmatched}")

    # ---- 1. per-segmentation means, with the new label ----------------------
    g = voxels.groupby("sample_segmentation")
    per = pd.DataFrame({
        "rs_selected_mean": g.rs_selected.mean(),
        "rs_selected_STD": g.rs_selected.std(ddof=1),
        "rs_selected_n_voxels": g.rs_selected.size(),
        "I0_anchored_mean": g.i0_tissue_anchored.mean(),
        "I0_anchored_STD": g.i0_tissue_anchored.std(ddof=1),
        "I0_anchored_n_voxels": g.i0_tissue_anchored.size()}).reset_index()
    per.insert(0, "new_label", per.sample_segmentation.map(mapping))
    per["matched"] = per.new_label.notna()
    per["new_label"] = per.new_label.fillna(per.sample_segmentation)
    per = per.rename(columns={"sample_segmentation": "old_label"})
    rank = {o: i for i, o in enumerate(labels["Old labels"].astype(str))}
    per["_order"] = per.old_label.map(rank).fillna(9_999)
    per = per.sort_values("_order").drop(columns="_order")
    per = per[["new_label", "old_label", "rs_selected_mean", "rs_selected_STD",
               "rs_selected_n_voxels", "I0_anchored_mean", "I0_anchored_STD",
               "I0_anchored_n_voxels", "matched"]]

    # ---- 2. combined by label, pooled statistics ----------------------------
    g = voxels.groupby("label_or_id")
    combined = pd.DataFrame({
        "rs_selected_mean": g.rs_selected.mean(),
        "rs_selected_STD": g.rs_selected.std(ddof=1),
        "rs_selected_n_voxels": g.rs_selected.size(),
        "I0_anchored_mean": g.i0_tissue_anchored.mean(),
        "I0_anchored_STD": g.i0_tissue_anchored.std(ddof=1),
        "I0_anchored_n_voxels": g.i0_tissue_anchored.size(),
        "n_segmentations_combined": g.sample_segmentation.nunique()}).reset_index()
    order = []
    for lab in labels["New labels"].astype(str):
        if lab not in order:
            order.append(lab)
    order += [lab for lab in combined.label_or_id if lab not in order]
    combined = (combined.set_index("label_or_id").reindex(order).reset_index()
                .rename(columns={"label_or_id": "new_label"}))
    assert combined.rs_selected_n_voxels.sum() == len(voxels), "voxels lost"

    # ---- 3 and 4. organ x species tables -----------------------------------
    tissue = voxels[voxels.new_label.notna()].copy()
    parsed = tissue.new_label.apply(lambda s: pd.Series(split_label(s), index=["organ", "species"]))
    tissue[["organ", "species"]] = parsed
    extras = [o for o in sorted(tissue.organ.unique()) if o not in ORGANS]
    print(f"organs found beyond the requested six: {extras}")

    def organ_table(value):
        grouped = tissue.groupby(["organ", "species"])[value]
        stats = {key: (m, s, n) for key, m, s, n in
                 zip(grouped.mean().index, grouped.mean(),
                     grouped.std(ddof=1), grouped.size())}
        rows = []
        for organ in ORGANS + extras:
            row = {"organ": organ}
            for species in SPECIES:
                m, s, n = stats.get((organ, species), (np.nan, np.nan, 0))
                row[f"{species} mean"] = m
                row[f"{species} std"] = s
                row[f"{species} number of voxels"] = int(n)
            rows.append(row)
        columns = ["organ"] + [f"{sp} {k}" for sp in SPECIES
                               for k in ("mean", "std", "number of voxels")]
        table = pd.DataFrame(rows)[columns]
        counted = sum(table[f"{sp} number of voxels"].sum() for sp in SPECIES)
        assert counted == len(tissue), f"{value}: {counted} of {len(tissue)} voxels"
        return table

    rs_table = organ_table("rs_selected")
    i0_table = organ_table("i0_tissue_anchored")

    wb = load_workbook(MEANS)
    before = list(wb.sheetnames)
    for name in REBUILD:
        if name in wb.sheetnames:
            del wb[name]
    write(wb, "means_new_labels", per, [26, 28, 17, 17, 17, 17, 17, 17, 10])
    write(wb, "weighted_means_by_label", combined, [26, 17, 17, 17, 17, 17, 17, 15])
    write(wb, "rs_by_organ", rs_table, [14] + [15] * 9)
    write(wb, "i0_by_organ", i0_table, [14] + [15] * 9)
    wb.save(MEANS)

    print(f"\nsheets before: {before}")
    print(f"sheets after : {load_workbook(MEANS).sheetnames}")
    for name in KEEP:
        assert name in wb.sheetnames, f"{name} was lost"
    pd.set_option("display.width", 215)
    print("\n=== weighted_means_by_label ===")
    print(combined.round(5).to_string(index=False))
    print("\n=== rs_by_organ ===")
    print(rs_table.round(5).to_string(index=False))
    print("\n=== i0_by_organ ===")
    print(i0_table.round(5).to_string(index=False))
    print(f"\nelapsed {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
