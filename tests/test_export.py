"""Tests for saving segmentations to disk, sparse and dense.

What these protect:

* the folder layout the analysis scripts will depend on;
* sparse files holding **only** mask voxels - no background, no zero padding;
* dense volumes placing the original intensity inside the mask and 0 outside;
* the same mask being applied at every echo time;
* metadata sufficient to rebuild the volume without the dense files;
* saving never re-running the segmentation, and never altering it.
"""

import json

import numpy as np
import pytest

from echoviewer import EchoSeries, segment_volume
from echoviewer.export import (
    DENSE_DIRNAME,
    SPARSE_DIRNAME,
    export_all,
    export_sample,
    load_dense,
    load_mask,
    load_sparse,
    reconstruct_dense_from_sparse,
    segment_dirname,
)


@pytest.fixture
def exported(tmp_path, segmentable_tree):
    """Export one synthetic sample and hand back everything needed to check it."""
    series = EchoSeries.from_sample(segmentable_tree, "Syringes")
    segmentation = segment_volume(series.volume(0))
    output = tmp_path / "Segmentations"
    report = export_sample(series, segmentation, output)
    yield series, segmentation, output / "Syringes", report
    series.close()


# --- layout -----------------------------------------------------------------


def test_folder_structure_matches_the_specification(exported):
    _, segmentation, root, _ = exported
    assert (root / "metadata.json").is_file()
    assert (root / SPARSE_DIRNAME).is_dir()
    assert (root / DENSE_DIRNAME).is_dir()

    for segment in segmentation.segments:
        name = segment_dirname(segment.index)
        assert (root / SPARSE_DIRNAME / name).is_dir()
        assert (root / DENSE_DIRNAME / name).is_dir()
        assert (root / SPARSE_DIRNAME / name / "metadata.json").is_file()
        assert (root / DENSE_DIRNAME / name / "mask.npz").is_file()


def test_one_dense_volume_per_echo_time_named_after_the_source_folder(exported):
    series, segmentation, root, _ = exported
    for segment in segmentation.segments:
        folder = root / DENSE_DIRNAME / segment_dirname(segment.index)
        for echo in series.echotimes:
            assert (folder / echo.path.name / "volume.npz").is_file(), (
                f"missing dense volume for {echo.label}"
            )


def test_report_counts_what_it_wrote(exported):
    series, segmentation, _, report = exported
    assert report.segments == [s.index for s in segmentation.segments]
    assert len(report.echo_times) == len(series)
    assert report.bytes_written > 0
    assert all(path.exists() for path in report.files)


# --- sparse -----------------------------------------------------------------


def _sparse_path(root, index):
    folder = root / SPARSE_DIRNAME / segment_dirname(index)
    for name in ("voxels.parquet", "voxels.csv.gz"):
        if (folder / name).is_file():
            return folder / name
    raise AssertionError(f"no sparse file in {folder}")


def test_sparse_has_the_required_columns(exported):
    _, segmentation, root, _ = exported
    frame = load_sparse(_sparse_path(root, segmentation.segments[0].index))
    assert list(frame.columns) == ["x", "y", "z", "EchoTime", "intensity"]


def test_sparse_holds_every_voxel_at_every_echo_time(exported):
    series, segmentation, root, _ = exported
    segment = segmentation.segments[0]
    frame = load_sparse(_sparse_path(root, segment.index))
    assert len(frame) == segment.voxels * len(series)
    assert sorted(frame["EchoTime"].unique()) == pytest.approx(
        [e.value for e in series.echotimes]
    )


def test_sparse_contains_only_mask_voxels(exported):
    """The point of the sparse form: nothing outside the segmentation."""
    _, segmentation, root, _ = exported
    segment = segmentation.segments[0]
    mask = load_mask(root / DENSE_DIRNAME / segment_dirname(segment.index) / "mask.npz")
    frame = load_sparse(_sparse_path(root, segment.index))

    inside = mask[
        frame["z"].to_numpy(dtype=np.intp),
        frame["y"].to_numpy(dtype=np.intp),
        frame["x"].to_numpy(dtype=np.intp),
    ]
    assert inside.all(), "sparse table contains voxels outside the mask"

    # And every mask voxel is present, so nothing was dropped either.
    unique = {tuple(row) for row in frame[["z", "y", "x"]].to_numpy()}
    assert len(unique) == int(mask.sum()) == segment.voxels


def test_sparse_intensities_match_the_source_volume(exported):
    series, segmentation, root, _ = exported
    segment = segmentation.segments[0]
    frame = load_sparse(_sparse_path(root, segment.index))

    for echo_index in (0, len(series) - 1):
        echo = series.echotime(echo_index)
        data = series.volume(echo_index).data
        rows = frame[np.isclose(frame["EchoTime"], echo.value)]
        expected = data[
            rows["z"].to_numpy(dtype=np.intp),
            rows["y"].to_numpy(dtype=np.intp),
            rows["x"].to_numpy(dtype=np.intp),
        ]
        np.testing.assert_allclose(
            rows["intensity"].to_numpy(dtype=np.float32), expected, rtol=1e-5
        )


# --- dense ------------------------------------------------------------------


def test_dense_has_the_original_shape(exported):
    series, segmentation, root, _ = exported
    dense = load_dense(
        root / DENSE_DIRNAME / segment_dirname(1) / series.echotime(0).path.name / "volume.npz"
    )
    assert dense.shape == tuple(segmentation.source_shape)


def test_dense_is_intensity_inside_and_zero_outside(exported):
    series, segmentation, root, _ = exported
    for segment in segmentation.segments[:2]:
        name = segment_dirname(segment.index)
        mask = load_mask(root / DENSE_DIRNAME / name / "mask.npz")
        for echo_index in (0, len(series) - 1):
            echo = series.echotime(echo_index)
            dense = load_dense(root / DENSE_DIRNAME / name / echo.path.name / "volume.npz")
            source = series.volume(echo_index).data
            np.testing.assert_allclose(dense[mask], source[mask], rtol=1e-5)
            assert not dense[~mask].any(), "voxels outside the mask are not zero"


def test_the_same_mask_is_used_at_every_echo_time(exported):
    """The masks are echo-invariant, so every echo must occupy the same voxels."""
    series, segmentation, root, _ = exported
    name = segment_dirname(segmentation.segments[0].index)
    footprints = [
        load_dense(root / DENSE_DIRNAME / name / echo.path.name / "volume.npz") != 0
        for echo in series.echotimes
    ]
    for other in footprints[1:]:
        np.testing.assert_array_equal(footprints[0], other)


# --- metadata and reconstruction --------------------------------------------


def test_metadata_describes_the_original_volume(exported):
    series, segmentation, root, _ = exported
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    volume = series.volume(0)

    assert tuple(metadata["shape"]["dimensions"]) == tuple(volume.shape)
    assert metadata["spacing_mm"]["z"] == pytest.approx(volume.spacing[0])
    assert metadata["spacing_mm"]["y"] == pytest.approx(volume.spacing[1])
    assert metadata["spacing_mm"]["x"] == pytest.approx(volume.spacing[2])
    assert metadata["echo_times_ms"] == pytest.approx([e.value for e in series.echotimes])
    assert metadata["segmentation"]["computed_on_echo_time_ms"] == pytest.approx(
        series.echotime(0).value
    )
    assert metadata["shape"]["array_index_order"] == ["z", "y", "x"]


def test_segment_metadata_records_size_and_position(exported):
    _, segmentation, root, _ = exported
    segment = segmentation.segments[0]
    metadata = json.loads(
        (root / SPARSE_DIRNAME / segment_dirname(segment.index) / "metadata.json").read_text()
    )
    assert metadata["segment"]["voxels"] == segment.voxels
    assert metadata["segment"]["index"] == segment.index
    assert len(metadata["segment"]["centroid_zyx"]) == 3


def test_sparse_rebuilds_the_dense_volume(exported):
    """Sparse plus metadata is enough to restore full spatial context."""
    series, segmentation, root, _ = exported
    segment = segmentation.segments[0]
    name = segment_dirname(segment.index)
    echo = series.echotime(0)

    rebuilt = reconstruct_dense_from_sparse(
        _sparse_path(root, segment.index),
        root / SPARSE_DIRNAME / name / "metadata.json",
        echo.value,
    )
    stored = load_dense(root / DENSE_DIRNAME / name / echo.path.name / "volume.npz")
    np.testing.assert_allclose(rebuilt, stored, rtol=1e-4, atol=1e-3)


# --- the segmentation must not be touched -----------------------------------


def test_export_does_not_recompute_or_alter_the_segmentation(tmp_path, segmentable_tree):
    """Saving is a pure write: same object, same labels, no new segmentation."""
    import echoviewer.segmentation as segmentation_module

    series = EchoSeries.from_sample(segmentable_tree, "Syringes")
    segmentation = segment_volume(series.volume(0))
    before = segmentation.labels.copy()

    calls = []
    original = segmentation_module.segment_volume
    segmentation_module.segment_volume = lambda *a, **k: calls.append(1) or original(*a, **k)
    try:
        export_sample(series, segmentation, tmp_path / "out")
    finally:
        segmentation_module.segment_volume = original
        series.close()

    assert calls == [], "export re-ran the segmentation"
    np.testing.assert_array_equal(segmentation.labels, before)


def test_export_refuses_an_empty_segmentation(tmp_path, segmentable_tree):
    from dataclasses import replace

    series = EchoSeries.from_sample(segmentable_tree, "Syringes")
    empty = replace(segment_volume(series.volume(0)), segments=())
    with pytest.raises(ValueError, match="no segmentations"):
        export_sample(series, empty, tmp_path / "out")
    series.close()


def test_export_all_covers_every_sample(tmp_path, segmentable_tree):
    reports = export_all(segmentable_tree, tmp_path / "out")
    assert {r.sample for r in reports} == {"Syringes", "Pill1"}
    for report in reports:
        assert (report.root / "metadata.json").is_file()


def test_export_all_can_be_restricted(tmp_path, segmentable_tree):
    reports = export_all(segmentable_tree, tmp_path / "out", samples=["Pill1"])
    assert [r.sample for r in reports] == ["Pill1"]
    assert not (tmp_path / "out" / "Syringes").exists()


# --- preview images ---------------------------------------------------------


def test_preview_image_per_segmentation(exported):
    """A folder called Segmentation_3 needs a picture to be identifiable."""
    from echoviewer.preview import IMAGES_DIRNAME, preview_filename

    _, segmentation, root, _ = exported
    folder = root / IMAGES_DIRNAME
    assert folder.is_dir()
    for segment in segmentation.segments:
        path = folder / preview_filename("Syringes", segment.index)
        assert path.is_file(), f"no preview for {segment.name}"
        assert path.stat().st_size > 0
    assert len(list(folder.glob("*.png"))) == len(segmentation)


def test_images_sit_beside_the_data_folders(exported):
    from echoviewer.preview import IMAGES_DIRNAME

    _, _, root, _ = exported
    assert {p.name for p in root.iterdir() if p.is_dir()} == {
        SPARSE_DIRNAME,
        DENSE_DIRNAME,
        IMAGES_DIRNAME,
    }


def test_preview_uses_the_viewer_overlay_style():
    """The saved images and the screen must not drift apart."""
    from echoviewer import preview, ui

    assert ui.OVERLAY_RGB is preview.OVERLAY_RGB
    assert ui.OVERLAY_ALPHA == preview.OVERLAY_ALPHA == 0.5


def test_preview_shows_three_orthogonal_views_and_titles(segmentable_tree):
    from echoviewer.preview import render_segment_preview

    series = EchoSeries.from_sample(segmentable_tree, "Syringes")
    try:
        volume = series.volume(0)
        segmentation = segment_volume(volume)
        figure = render_segment_preview(
            volume, segmentation, 1, "Syringes", series.echotime(0).label
        )
        titles = [axes.get_title() for axes in figure.axes]
        assert len(titles) == 3
        assert any("Axial" in t for t in titles)
        assert any("Coronal" in t for t in titles)
        assert any("Sagittal" in t for t in titles)

        # Sample, segmentation number and echo time all appear in the title.
        suptitle = figure._suptitle.get_text()
        assert "Syringes" in suptitle
        assert "Segmentation 1" in suptitle
        assert series.echotime(0).label in suptitle
    finally:
        series.close()


def test_preview_overlays_purple_at_half_opacity(segmentable_tree):
    """Two images per panel: greyscale underneath, transparent purple on top."""
    import numpy as np

    from echoviewer.preview import OVERLAY_ALPHA, OVERLAY_RGB, render_segment_preview

    series = EchoSeries.from_sample(segmentable_tree, "Syringes")
    try:
        volume = series.volume(0)
        segmentation = segment_volume(volume)
        figure = render_segment_preview(volume, segmentation, 1, "Syringes", "TE = 8 ms")

        axes = figure.axes[0]
        assert len(axes.images) == 2, "expected a greyscale layer and an overlay"
        base, overlay = axes.images
        assert base.get_cmap().name == "gray"
        assert overlay.get_zorder() > base.get_zorder()

        rgba = np.asarray(overlay.get_array())
        painted = rgba[..., 3] > 0
        assert painted.any()
        np.testing.assert_allclose(rgba[painted][0, :3], OVERLAY_RGB, atol=1e-6)
        assert rgba[..., 3].max() == pytest.approx(OVERLAY_ALPHA)
    finally:
        series.close()


def test_preview_picks_the_slice_with_most_mask_voxels(segmentable_tree):
    from echoviewer.preview import _busiest_slice

    from dicomview.volume import View

    series = EchoSeries.from_sample(segmentable_tree, "Syringes")
    try:
        segmentation = segment_volume(series.volume(0))
        mask = segmentation.labels == 1
        for view, axes in (
            (View.AXIAL, (1, 2)),
            (View.CORONAL, (0, 2)),
            (View.SAGITTAL, (0, 1)),
        ):
            index = _busiest_slice(mask, view)
            counts = mask.sum(axis=axes)
            assert counts[index] == counts.max()
    finally:
        series.close()


def test_preview_refuses_a_mismatched_volume(segmentable_tree):
    from dataclasses import replace

    from echoviewer.preview import render_segment_preview

    series = EchoSeries.from_sample(segmentable_tree, "Syringes")
    try:
        volume = series.volume(0)
        segmentation = replace(
            segment_volume(volume), source_shape=(9, 9, 9)
        )
        with pytest.raises(ValueError, match="does not fit"):
            render_segment_preview(volume, segmentation, 1, "Syringes", "TE = 8 ms")
    finally:
        series.close()


def test_images_can_be_written_without_the_data(tmp_path, segmentable_tree):
    from echoviewer.preview import IMAGES_DIRNAME

    series = EchoSeries.from_sample(segmentable_tree, "Syringes")
    try:
        segmentation = segment_volume(series.volume(0))
        report = export_sample(
            series,
            segmentation,
            tmp_path / "out",
            write_sparse=False,
            write_dense=False,
        )
        root = tmp_path / "out" / "Syringes"
        assert list((root / IMAGES_DIRNAME).glob("*.png"))
        assert report.echo_times == []  # the echo walk was skipped entirely
    finally:
        series.close()


def test_sparse_only_and_dense_only(tmp_path, segmentable_tree):
    series = EchoSeries.from_sample(segmentable_tree, "Syringes")
    segmentation = segment_volume(series.volume(0))
    try:
        sparse_root = tmp_path / "sparse" / "Syringes"
        export_sample(series, segmentation, tmp_path / "sparse", write_dense=False)
        assert (sparse_root / SPARSE_DIRNAME).is_dir()
        assert not (sparse_root / DENSE_DIRNAME).exists()

        dense_root = tmp_path / "dense" / "Syringes"
        export_sample(series, segmentation, tmp_path / "dense", write_sparse=False)
        assert (dense_root / DENSE_DIRNAME).is_dir()
        assert not any((dense_root / SPARSE_DIRNAME).glob("*/voxels*"))
    finally:
        series.close()
