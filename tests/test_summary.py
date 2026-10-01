"""Tests for the mean-intensity workbook.

The contract: one sheet per sample, ``EchoTime`` as the first column and one
column per segmentation, means taken over mask voxels **only**, echo times
preserved exactly, blanks rather than zeros where there is nothing to average,
and no segmentation run anywhere in the process.
"""

import numpy as np
import pytest

from echoviewer import EchoSeries, segment_volume
from echoviewer.export import export_sample, load_mask, segment_dirname
from echoviewer.summary import (
    DEFAULT_WORKBOOK,
    build_tables,
    export_mean_intensities,
    load_saved_masks,
    mean_intensity_table,
    write_workbook,
)


@pytest.fixture
def exported_tree(tmp_path, segmentable_tree):
    """Run the real export once, so the summary reads genuinely saved masks."""
    output = tmp_path / "Segmentations"
    for sample in ("Syringes", "Pill1"):
        series = EchoSeries.from_sample(segmentable_tree, sample)
        try:
            export_sample(
                series,
                segment_volume(series.volume(0)),
                output,
                write_images=False,
            )
        finally:
            series.close()
    return segmentable_tree, output


# --- reading the saved masks ------------------------------------------------


def test_masks_are_loaded_from_disk(exported_tree):
    _, output = exported_tree
    masks = load_saved_masks(output / "Syringes")
    assert masks
    assert all(mask.dtype == bool for mask in masks.values())
    assert sorted(masks) == list(range(1, len(masks) + 1))


def test_missing_masks_give_an_empty_mapping(tmp_path):
    assert load_saved_masks(tmp_path / "nothing") == {}


def test_build_tables_needs_an_export(tmp_path, segmentable_tree):
    with pytest.raises(FileNotFoundError):
        build_tables(segmentable_tree, tmp_path / "empty")


# --- table shape ------------------------------------------------------------


def test_first_column_is_echotime_then_one_column_per_segmentation(exported_tree):
    volumes, output = exported_tree
    masks = load_saved_masks(output / "Syringes")
    series = EchoSeries.from_sample(volumes, "Syringes")
    try:
        table = mean_intensity_table(series, masks)
    finally:
        series.close()

    assert table.columns[0] == "EchoTime"
    assert list(table.columns[1:]) == [segment_dirname(i) for i in sorted(masks)]
    assert len(table) == len(series)


def test_echo_times_are_preserved_exactly_and_sorted(exported_tree):
    volumes, output = exported_tree
    series = EchoSeries.from_sample(volumes, "Syringes")
    try:
        table = mean_intensity_table(series, load_saved_masks(output / "Syringes"))
    finally:
        series.close()

    expected = sorted(e.value for e in series.echotimes)
    assert table["EchoTime"].tolist() == pytest.approx(expected)
    assert table["EchoTime"].is_monotonic_increasing


# --- the values themselves --------------------------------------------------


def test_values_are_the_mean_of_mask_voxels_only(exported_tree):
    """The check that matters: np.mean(volume[mask]), computed independently."""
    volumes, output = exported_tree
    masks = load_saved_masks(output / "Syringes")
    series = EchoSeries.from_sample(volumes, "Syringes")
    try:
        table = mean_intensity_table(series, masks)
        for echo_index in range(len(series)):
            echo = series.echotime(echo_index)
            data = series.volume(echo_index).data
            row = table[np.isclose(table["EchoTime"], echo.value)].iloc[0]
            for index, mask in masks.items():
                expected = float(np.mean(data[mask]))
                assert row[segment_dirname(index)] == pytest.approx(expected, rel=1e-6)
    finally:
        series.close()


def test_zeros_outside_the_mask_do_not_contribute(exported_tree):
    """Averaging the zero-filled dense volume would be wrong by a large factor."""
    volumes, output = exported_tree
    masks = load_saved_masks(output / "Syringes")
    series = EchoSeries.from_sample(volumes, "Syringes")
    try:
        table = mean_intensity_table(series, masks)
        data = series.volume(0).data
        mask = masks[1]

        inside_only = float(np.mean(data[mask]))
        with_zeros = float(np.mean(np.where(mask, data, 0.0)))
        reported = float(table.iloc[0][segment_dirname(1)])

        assert reported == pytest.approx(inside_only, rel=1e-6)
        assert reported != pytest.approx(with_zeros, rel=1e-3)
        assert with_zeros < inside_only  # diluted by the background
    finally:
        series.close()


def test_agrees_with_the_sparse_export(exported_tree):
    """Cross-check against a different saved artefact of the same pipeline."""
    from echoviewer.export import SPARSE_DIRNAME, load_sparse

    volumes, output = exported_tree
    folder = output / "Syringes" / SPARSE_DIRNAME / segment_dirname(1)
    sparse_file = next(
        p for p in folder.iterdir() if p.name.startswith("voxels.")
    )
    frame = load_sparse(sparse_file)
    by_echo = frame.groupby("EchoTime")["intensity"].mean()

    series = EchoSeries.from_sample(volumes, "Syringes")
    try:
        table = mean_intensity_table(series, load_saved_masks(output / "Syringes"))
    finally:
        series.close()

    for _, row in table.iterrows():
        assert row[segment_dirname(1)] == pytest.approx(
            float(by_echo[row["EchoTime"]]), rel=1e-4
        )


def test_empty_mask_gives_nan_not_zero(exported_tree):
    """0 is a plausible intensity, so it must not stand in for "no data"."""
    volumes, output = exported_tree
    masks = load_saved_masks(output / "Syringes")
    masks[max(masks) + 1] = np.zeros_like(next(iter(masks.values())))

    series = EchoSeries.from_sample(volumes, "Syringes")
    try:
        table = mean_intensity_table(series, masks)
    finally:
        series.close()

    column = segment_dirname(max(masks))
    assert table[column].isna().all()
    assert not (table[column] == 0).any()


def test_mismatched_mask_gives_nan(exported_tree):
    volumes, output = exported_tree
    masks = {1: np.ones((3, 3, 3), dtype=bool)}
    series = EchoSeries.from_sample(volumes, "Syringes")
    try:
        table = mean_intensity_table(series, masks)
    finally:
        series.close()
    assert table[segment_dirname(1)].isna().all()


def test_no_masks_is_an_error(exported_tree):
    volumes, _ = exported_tree
    series = EchoSeries.from_sample(volumes, "Syringes")
    try:
        with pytest.raises(ValueError, match="no saved masks"):
            mean_intensity_table(series, {})
    finally:
        series.close()


# --- the workbook -----------------------------------------------------------


def test_workbook_has_one_sheet_per_sample(exported_tree, tmp_path):
    import pandas as pd

    volumes, output = exported_tree
    path, tables = export_mean_intensities(volumes, output, workbook=tmp_path / "book.xlsx")

    assert path.is_file()
    sheets = pd.ExcelFile(path, engine="openpyxl").sheet_names
    assert set(sheets) == set(tables) == {"Syringes", "Pill1"}


def test_workbook_reloads_with_the_same_numbers(exported_tree, tmp_path):
    import pandas as pd

    volumes, output = exported_tree
    path, tables = export_mean_intensities(volumes, output, workbook=tmp_path / "book.xlsx")

    reloaded = pd.read_excel(path, sheet_name="Syringes", engine="openpyxl")
    expected = tables["Syringes"]
    assert list(reloaded.columns) == list(expected.columns)
    for column in expected.columns:
        np.testing.assert_allclose(
            reloaded[column].to_numpy(dtype=float),
            expected[column].to_numpy(dtype=float),
            rtol=1e-6,
        )


def test_default_workbook_lands_in_the_segmentations_folder(exported_tree):
    volumes, output = exported_tree
    path, _ = export_mean_intensities(volumes, output)
    assert path == output / DEFAULT_WORKBOOK
    assert path.is_file()


def test_suffix_is_added(exported_tree, tmp_path):
    volumes, output = exported_tree
    tables = build_tables(volumes, output)
    path = write_workbook(tables, tmp_path / "no_suffix")
    assert path.suffix == ".xlsx"
    assert path.is_file()


def test_samples_can_be_restricted(exported_tree, tmp_path):
    volumes, output = exported_tree
    _, tables = export_mean_intensities(
        volumes, output, workbook=tmp_path / "one.xlsx", samples=["Pill1"]
    )
    assert list(tables) == ["Pill1"]


# --- nothing is recomputed or altered ---------------------------------------


def test_summary_never_segments(exported_tree, tmp_path):
    import echoviewer.segmentation as segmentation_module

    volumes, output = exported_tree
    calls = []
    original = segmentation_module.segment_volume
    segmentation_module.segment_volume = lambda *a, **k: calls.append(1) or original(*a, **k)
    try:
        export_mean_intensities(volumes, output, workbook=tmp_path / "book.xlsx")
    finally:
        segmentation_module.segment_volume = original

    assert calls == [], "the summary step re-ran the segmentation"


def test_summary_does_not_touch_the_saved_data(exported_tree, tmp_path):
    """Purely an analysis pass: every exported file must be left as it was."""
    volumes, output = exported_tree
    before = {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }

    export_mean_intensities(volumes, output, workbook=tmp_path / "book.xlsx")

    after = {
        path: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    assert before == after


def test_masks_are_unchanged_after_summarising(exported_tree, tmp_path):
    volumes, output = exported_tree
    mask_path = output / "Syringes" / "volumes" / segment_dirname(1) / "mask.npz"
    before = load_mask(mask_path).copy()
    export_mean_intensities(volumes, output, workbook=tmp_path / "book.xlsx")
    np.testing.assert_array_equal(load_mask(mask_path), before)
