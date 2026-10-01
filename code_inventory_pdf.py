#!/usr/bin/env python3
"""Build a PDF inventory of every Python file in both dicom_viewer trees.

Purposes are read from each module's docstring at run time, so the document
cannot drift from the code; the grouping into sections is the only editorial
part. Read-only: the script imports nothing from the modules it describes and
writes a single PDF.
"""

import ast
import sys
import time
from pathlib import Path
from config import DICOMVIEW_HOME, REPO  # noqa: E402

PROJECT = REPO
ORIGINAL = DICOMVIEW_HOME
TARGET = PROJECT / "code_inventory.pdf"

#: section title -> ordered file paths, relative to PROJECT
SECTIONS = [
    ("echoviewer — echo-time viewer and analysis engine", [
        "echoviewer/__init__.py", "echoviewer/__main__.py", "echoviewer/_bootstrap.py",
        "echoviewer/dataset.py", "echoviewer/series.py", "echoviewer/ui.py",
        "echoviewer/segmentation.py", "echoviewer/preview.py", "echoviewer/export.py",
        "echoviewer/summary.py", "echoviewer/fitting.py", "echoviewer/relaxometry.py",
        "echoviewer/model_selected.py", "echoviewer/diagnostics.py"]),
    ("Top-level runners", [
        "run_echo_viewer.py", "run_diagnostics.py", "run_model_selected.py",
        "run_normalization.py", "run_save_normalized.py", "run_high_rs_artifact.py"]),
    ("data_treatment — normalisation and feature preparation", [
        "data_treatment/norm_comparison.py", "data_treatment/overlap_2d.py",
        "data_treatment/overlap_prepost.py", "data_treatment/residual_feature_space.py",
        "data_treatment/verify_labellight.py",
        "data_treatment/apply_labellight_normalization.py",
        "data_treatment/normalize_pill2_volume.py",
        "data_treatment/prepare_pill2_testing.py", "data_treatment/validate_fit.py",
        "data_treatment/fit_report.py"]),
    ("data_treatment — feature diagnostics", [
        "data_treatment/b_parameter_diagnostic.py",
        "data_treatment/b_normalization_transfer.py",
        "data_treatment/derived_feature_transfer.py",
        "data_treatment/shape_parameter_diagnostic.py",
        "data_treatment/class_feature_plots.py",
        "data_treatment/feature_space_train_vs_pill2.py",
        "data_treatment/foreground_diagnosis.py", "data_treatment/show_pill2_extent.py"]),
    ("data_treatment — classification", [
        "data_treatment/train_baseline_classifier.py",
        "data_treatment/pill2_complete_segmentation_eval.py",
        "data_treatment/complete_dAIC_and_compare.py",
        "data_treatment/classifier_benchmark.py",
        "data_treatment/classifier_benchmark_extension.py",
        "data_treatment/classifier_benchmark_summary.py"]),
    ("data_treatment — metrics", [
        "data_treatment/auc_metrics.py", "data_treatment/iou_metrics.py",
        "data_treatment/segmentation_metrics.py", "data_treatment/organise_metrics.py",
        "data_treatment/write_metrics_summary_md.py",
        "data_treatment/iou_corrected_labels.py",
        "data_treatment/pill2_metrics_corrected_labels.py"]),
    ("data_treatment — rankings and workbooks", [
        "data_treatment/rank_top3_models.py",
        "data_treatment/best_models_by_class_xlsx.py",
        "data_treatment/rank_corrected_labels.py",
        "data_treatment/best_model_matrix_corrected_labels.py",
        "data_treatment/macro_ranking_sheets.py", "data_treatment/class_ranking_sheet.py",
        "data_treatment/weighted_summary_workbook.py",
        "data_treatment/bump_chart_workbook.py", "data_treatment/results_workbooks.py",
        "data_treatment/results_workbooks_subsample.py"]),
    ("data_treatment — figures and volumes", [
        "data_treatment/peak_slice_montage.py", "data_treatment/peak_slice_vector_svg.py",
        "data_treatment/gmm_probability_volume.py",
        "data_treatment/gmm_all_classes_volume.py",
        "data_treatment/gmm_all_classes_overview_svg.py",
        "data_treatment/all_models_probability_volumes.py",
        "data_treatment/all_models_axial_overviews.py",
        "data_treatment/segmentation_overlays.py", "data_treatment/rs_map_overview.py",
        "data_treatment/echo_first_last_overview.py", "data_treatment/coronal_montage.py",
        "data_treatment/render_volume_3d.py", "data_treatment/pill1_threshold_mask.py",
        "data_treatment/make_colorbars.py", "data_treatment/__init__.py"]),
    ("human — human and mouse intestine analysis", [
        "human/fit_and_normalise_human.py", "human/tissue_anchored_normalisation.py",
        "human/rebuild_label_sheets.py"]),
    ("tests", [
        "tests/conftest.py", "tests/test_echoviewer.py", "tests/test_fitting.py",
        "tests/test_segmentation.py", "tests/test_export.py", "tests/test_summary.py"]),
]

#: only where a module has no docstring
FALLBACK = {
    "data_treatment/__init__.py": "Package marker (empty).",
    "data_treatment/write_metrics_summary_md.py": "Write Metrics/summary.md.",
}


def purpose(path: Path, key: str) -> str:
    try:
        doc = ast.get_docstring(ast.parse(path.read_text(errors="ignore")))
    except SyntaxError:
        doc = None
    if doc:
        first = doc.strip().splitlines()[0].strip()
        if first:
            return first
    return FALLBACK.get(key, "—")


def main() -> int:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (KeepTogether, Paragraph, SimpleDocTemplate, Spacer,
                                    Table, TableStyle)

    started = time.time()
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["Normal"], fontName="Helvetica",
                          fontSize=7.6, leading=9.4, alignment=TA_LEFT)
    mono = ParagraphStyle("mono", parent=body, fontName="Courier", fontSize=6.9,
                          leading=8.6, textColor=colors.HexColor("#333333"))
    name = ParagraphStyle("name", parent=body, fontName="Helvetica-Bold")
    head = ParagraphStyle("head", parent=styles["Heading2"], fontName="Helvetica-Bold",
                          fontSize=10.5, spaceBefore=12, spaceAfter=5,
                          textColor=colors.HexColor("#1F3864"))
    note = ParagraphStyle("note", parent=body, fontSize=7.6, leading=10,
                          textColor=colors.HexColor("#555555"))

    def table(rows):
        data = [[Paragraph("File", name), Paragraph("Purpose", name),
                 Paragraph("Path", name)]]
        for file_name, text, rel in rows:
            data.append([Paragraph(file_name, body), Paragraph(text, body),
                         Paragraph(rel, mono)])
        t = Table(data, colWidths=[42 * mm, 78 * mm, 63 * mm], repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3864")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#BBBBBB")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#F4F6FA")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]))
        return t

    story = []
    project_files = [p for p in PROJECT.rglob("*.py") if "__pycache__" not in p.parts]
    original_files = [p for p in ORIGINAL.rglob("*.py") if "__pycache__" not in p.parts]
    lines = sum(len(p.read_text(errors="ignore").splitlines())
                for p in project_files + original_files)

    story.append(Paragraph(f"Code inventory — {PROJECT.name}",
                           ParagraphStyle("title", parent=styles["Title"],
                                          fontSize=17, spaceAfter=4)))
    story.append(Paragraph(
        f"{len(project_files) + len(original_files)} Python files, {lines:,} lines, "
        f"across two trees. Generated {time.strftime('%Y-%m-%d')}. Each purpose is the "
        f"first line of the module's own docstring, read from the file at build time.",
        note))
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        f"1. {PROJECT.name} — {len(project_files)} files",
        ParagraphStyle("h1", parent=head, fontSize=12.5, textColor=colors.black)))
    covered = set()
    for title, keys in SECTIONS:
        rows = []
        for key in keys:
            path = PROJECT / key
            if not path.is_file():
                continue
            covered.add(key)
            rows.append((path.name, purpose(path, key), key))
        if rows:
            story.append(KeepTogether([Paragraph(title, head), table(rows)]))

    missing = sorted(str(p.relative_to(PROJECT)) for p in project_files
                     if str(p.relative_to(PROJECT)) not in covered)
    if missing:
        rows = [((PROJECT / k).name, purpose(PROJECT / k, k), k) for k in missing]
        story.append(KeepTogether([Paragraph("Other files", head), table(rows)]))

    story.append(Spacer(1, 10))
    story.append(Paragraph(
        f"2. {ORIGINAL.name} — the original viewer, {len(original_files)} files",
        ParagraphStyle("h1b", parent=head, fontSize=12.5, textColor=colors.black)))
    rows = [(p.name, purpose(p, str(p.relative_to(ORIGINAL))),
             str(p.relative_to(ORIGINAL)))
            for p in sorted(original_files)]
    story.append(table(rows))

    doc = SimpleDocTemplate(str(TARGET), pagesize=A4,
                            leftMargin=14 * mm, rightMargin=14 * mm,
                            topMargin=13 * mm, bottomMargin=13 * mm,
                            title=f"Code inventory - {PROJECT.name}",
                            author="generated from module docstrings")

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(14 * mm, 8 * mm, "dicom_viewer code inventory")
        canvas.drawRightString(A4[0] - 14 * mm, 8 * mm, f"page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    size = TARGET.stat().st_size
    print(f"saved {TARGET}")
    print(f"{len(project_files)} + {len(original_files)} files, {lines:,} lines, "
          f"{size/1000:.0f} kB")
    if missing:
        print(f"files not in a curated section, listed under 'Other files': {missing}")
    print(f"elapsed {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
