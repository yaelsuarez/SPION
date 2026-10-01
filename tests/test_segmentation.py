"""Tests for the 3D segmentation and its integration into the viewer.

The behavioural contract these protect:

* the segmentation algorithm runs **once per sample**, never on an echo,
  slice, plane or checkbox change;
* the masks are echo-time invariant and applied at the same voxel coordinates;
* the overlay is purple at 50% opacity, drawn *over* the greyscale rather than
  replacing it, and stays aligned in all three planes.
"""

import numpy as np
import pytest

from echoviewer.segmentation import (
    AUTO_PRESET,
    PRESETS,
    SegmentationParams,
    binarize_volume,
    segment_volume,
    to_gray_volume,
)

from dicomview.volume import View, Volume


# --- the 3D pipeline --------------------------------------------------------


def test_gray_normalisation_uses_one_global_maximum():
    """Per-slice scaling is what makes slice-by-slice segmentation inconsistent."""
    data = np.zeros((3, 4, 4), dtype=np.float32)
    data[0] = 100.0
    data[1] = 200.0
    data[2] = 50.0
    gray = to_gray_volume(data)
    # One scale for the whole volume: the brightest slice reaches 255 and the
    # others stay proportionally below it.
    assert gray[1].max() == pytest.approx(255.0)
    assert gray[0].max() == pytest.approx(127.5)
    assert gray[2].max() == pytest.approx(63.75)


def test_gray_handles_empty_and_non_finite():
    assert to_gray_volume(np.zeros((2, 2, 2))).max() == 0
    data = np.full((2, 2, 2), np.nan, dtype=np.float32)
    assert np.isfinite(to_gray_volume(data)).all()


def test_segments_are_coherent_through_the_stack(blobs_volume):
    """One physical object must get one index on every slice it appears in."""
    segmentation = segment_volume(blobs_volume)
    assert len(segmentation) == 3

    for segment in segmentation.segments:
        present = [
            k for k in range(blobs_volume.shape[0])
            if (segmentation.labels[k] == segment.index).any()
        ]
        # Each bar runs the whole length of the stack.
        assert present == list(range(blobs_volume.shape[0])), (
            f"{segment.name} is not continuous through the stack"
        )


def test_regions_do_not_overlap(blobs_volume):
    segmentation = segment_volume(blobs_volume)
    total = sum(s.voxels for s in segmentation.segments)
    labelled = int((segmentation.labels > 0).sum())
    assert total == labelled, "labels overlap or metadata disagrees with the map"


def test_labels_are_numbered_from_one_without_gaps(blobs_volume):
    segmentation = segment_volume(blobs_volume)
    present = sorted(np.unique(segmentation.labels))
    assert present == [0] + [s.index for s in segmentation.segments]
    assert [s.index for s in segmentation.segments] == list(range(1, len(segmentation) + 1))


def test_small_regions_are_dropped(blobs_volume):
    """min_voxels must remove speckle without removing the real bars."""
    params = SegmentationParams(method="otsu", min_voxels=10**6, erode_voxels=0)
    assert len(segment_volume(blobs_volume, params=params)) == 0


def test_erosion_shrinks_regions_but_never_deletes_them(blobs_volume):
    without = segment_volume(
        blobs_volume, params=SegmentationParams(method="otsu", erode_voxels=0)
    )
    with_erosion = segment_volume(
        blobs_volume, params=SegmentationParams(method="otsu", erode_voxels=1)
    )
    assert len(with_erosion) == len(without)
    for eroded, plain in zip(with_erosion.segments, without.segments):
        assert 0 < eroded.voxels < plain.voxels


def test_binarize_rejects_unknown_method(blobs_volume):
    with pytest.raises(ValueError, match="Unknown threshold method"):
        binarize_volume(
            to_gray_volume(blobs_volume.data), SegmentationParams(method="nonsense")
        )


def test_empty_volume_yields_no_segments_not_an_error():
    volume = Volume(data=np.zeros((4, 8, 8), dtype=np.float32), spacing=(1.0, 1.0, 1.0))
    segmentation = segment_volume(volume)
    assert len(segmentation) == 0
    assert segmentation.warnings


def test_label_volume_slices_exactly_like_the_image(blobs_volume):
    """The overlay is extracted by the viewer's own Volume.extract."""
    segmentation = segment_volume(blobs_volume)
    label_volume = segmentation.as_volume()
    for view in View:
        assert label_volume.n_slices(view) == blobs_volume.n_slices(view)
        index = blobs_volume.n_slices(view) // 2
        assert (
            label_volume.extract(view, index).image.shape
            == blobs_volume.extract(view, index).image.shape
        )


def test_matches_refuses_a_differently_shaped_volume(blobs_volume):
    segmentation = segment_volume(blobs_volume)
    assert segmentation.matches(blobs_volume)
    other = Volume(data=np.zeros((5, 8, 8), dtype=np.float32), spacing=(1.0, 1.0, 1.0))
    assert not segmentation.matches(other)


def test_presets_all_run(blobs_volume):
    for name, params in PRESETS.items():
        result = segment_volume(blobs_volume, params=params, preset=name)
        assert result.source_shape == tuple(blobs_volume.shape)


# --- compartments inside one connected object -------------------------------


def _banded_volume(levels, noise=2.0, seed=0):
    """A single bar whose length is divided into plateaus of the given levels."""
    rows = 20 * len(levels)
    data = np.zeros((10, rows, 20), dtype=np.float32)
    for position, level in enumerate(levels):
        data[2:8, 20 * position:20 * (position + 1), 5:15] = level
    rng = np.random.default_rng(seed)
    data += rng.normal(0.0, noise, data.shape).astype(np.float32)
    return Volume(data=np.clip(data, 0, None), spacing=(1.0, 1.0, 1.0))


def _ramp_volume(low=200.0, high=900.0, noise=2.0, seed=0):
    """A single bar whose brightness rises smoothly along its length."""
    rows = 80
    data = np.zeros((10, rows, 20), dtype=np.float32)
    ramp = np.linspace(low, high, rows, dtype=np.float32)
    data[2:8, :, 5:15] = ramp[None, :, None]
    rng = np.random.default_rng(seed)
    data += rng.normal(0.0, noise, data.shape).astype(np.float32)
    return Volume(data=np.clip(data, 0, None), spacing=(1.0, 1.0, 1.0))


def test_touching_compartments_are_separated(blobs_volume):
    """The Pill2 report: four compartments in one object came out as one mask."""
    volume = _banded_volume([900.0, 700.0, 500.0, 320.0])
    segmentation = segment_volume(volume)
    assert len(segmentation) == 4, (
        f"expected four compartments, got {[s.voxels for s in segmentation.segments]}"
    )

    from echoviewer.segmentation import to_gray_volume

    gray = to_gray_volume(volume.data)
    means = sorted(float(gray[segmentation.labels == s.index].mean())
                   for s in segmentation.segments)
    # Distinct, and in the order the plateaus were built.
    assert all(b - a > 10 for a, b in zip(means, means[1:]))


def test_a_smooth_gradient_is_not_carved_up():
    """Guards the fix: the syringes fade along their length and must stay whole.

    Their parts would differ by 25-29 grey levels if split - more than the
    Pill2 compartments do - so mean difference alone cannot reject them.
    """
    segmentation = segment_volume(_ramp_volume())
    assert len(segmentation) == 1, (
        f"a smooth ramp was split into {len(segmentation)} parts"
    )


def test_uniform_object_is_not_split():
    segmentation = segment_volume(_banded_volume([800.0, 800.0, 800.0]))
    assert len(segmentation) == 1


def test_band_splitting_can_be_switched_off():
    params = SegmentationParams(method="otsu", split_bands=False, erode_voxels=0)
    volume = _banded_volume([900.0, 700.0, 500.0, 320.0])
    assert len(segment_volume(volume, params=params)) == 1


def test_dim_material_is_pulled_into_its_region():
    """Pill2's last band sits under the global Otsu level and was being lost.

    The mechanism is what matters: material dimmer than the threshold, but
    attached to a region that cleared it, has to end up inside that region.
    On the real capsule this extends it from rows 132-226 to 131-235.
    """
    volume = _banded_volume([900.0, 700.0, 500.0, 260.0])
    without = segment_volume(
        volume, params=SegmentationParams(method="otsu", hysteresis_ratio=0.0, split_bands=False)
    )
    with_growth = segment_volume(
        volume, params=SegmentationParams(method="otsu", hysteresis_ratio=0.6, split_bands=False)
    )
    grown = sum(s.voxels for s in with_growth.segments)
    plain = sum(s.voxels for s in without.segments)
    assert grown > plain, "growth captured no extra material"
    # And it did not invent a new object to hold it.
    assert len(with_growth) == len(without) == 1


def test_growth_never_merges_separate_objects(blobs_volume):
    """The 3D phantom regressed when a lower threshold bridged ring and wedges."""
    plain = segment_volume(
        blobs_volume, params=SegmentationParams(method="otsu", hysteresis_ratio=0.0)
    )
    grown = segment_volume(
        blobs_volume, params=SegmentationParams(method="otsu", hysteresis_ratio=0.6)
    )
    assert len(grown) == len(plain) == 3


def test_step_sharpness_tells_a_step_from_a_ramp():
    from echoviewer.segmentation import _step_sharpness

    # Two halves of a step: the difference survives at the interface.
    gray = np.zeros((6, 20, 6), dtype=np.float32)
    gray[:, :10] = 200.0
    gray[:, 10:] = 100.0
    labels = np.zeros(gray.shape, dtype=np.int32)
    labels[:, :10] = 1
    labels[:, 10:] = 2
    assert _step_sharpness(gray, labels, [1, 2]) > 0.9

    # Same two halves of a ramp: neighbouring values are nearly equal.
    ramp = np.zeros((6, 20, 6), dtype=np.float32)
    ramp[:] = np.linspace(100.0, 200.0, 20, dtype=np.float32)[None, :, None]
    assert _step_sharpness(ramp, labels, [1, 2]) < 0.3


# --- viewer integration -----------------------------------------------------


@pytest.fixture
def viewer(qt_app, segmentable_tree):
    """A viewer on a synthetic tree that actually contains segmentable bars."""
    from echoviewer import ui as ui_module

    window = ui_module.EchoViewerWindow(segmentable_tree, sample="Syringes")
    yield window
    window.close()


def test_the_fixture_really_produces_masks(viewer):
    """Guards the tests below: without masks they would pass vacuously."""
    assert len(viewer._segmentation) >= 2
    assert len(viewer._segment_checks) == len(viewer._segmentation)


def test_segmentation_runs_once_when_a_sample_is_opened(viewer):
    assert viewer.segmentation_runs == 1
    assert viewer._segmentation is not None


def test_navigation_never_recomputes_segmentation(viewer):
    """The central requirement: masks are computed once and then only reused."""
    baseline = viewer.segmentation_runs

    for index in range(len(viewer._series)):      # every echo time
        viewer._set_echo(index)
    for plane in range(3):                        # every plane
        viewer._view_combo.setCurrentIndex(plane)
        limit = viewer._slice_slider.maximum()
        for slice_index in (0, limit // 2, limit):
            viewer._set_slice(slice_index)
    viewer._show_segmentation_check.setChecked(False)   # master toggle
    viewer._show_segmentation_check.setChecked(True)
    for check in viewer._segment_checks:                # individual masks
        check.setChecked(not check.isChecked())
    viewer._set_all_segments(True)

    assert viewer.segmentation_runs == baseline


def test_masks_are_identical_across_echo_times(viewer):
    """Same voxels, whatever the echo time."""
    reference = viewer._segmentation.labels.copy()
    for index in range(len(viewer._series)):
        viewer._set_echo(index)
        np.testing.assert_array_equal(viewer._segmentation.labels, reference)


def test_checkbox_states_survive_an_echo_change(viewer):
    if len(viewer._segment_checks) < 2:
        pytest.skip("needs at least two masks")
    viewer._segment_checks[0].setChecked(True)
    viewer._segment_checks[1].setChecked(False)
    viewer._show_segmentation_check.setChecked(True)

    viewer._set_echo(len(viewer._series) - 1)

    assert viewer._segment_checks[0].isChecked()
    assert not viewer._segment_checks[1].isChecked()
    assert viewer._show_segmentation_check.isChecked()


def test_master_toggle_hides_overlay_without_clearing_selections(viewer):
    viewer._set_all_segments(True)
    selected_before = [c.isChecked() for c in viewer._segment_checks]

    viewer._show_segmentation_check.setChecked(False)
    assert viewer._selected_segment_indices() == set()
    assert [c.isChecked() for c in viewer._segment_checks] == selected_before

    viewer._show_segmentation_check.setChecked(True)
    assert viewer._selected_segment_indices()


def test_individual_checkboxes_select_individual_masks(viewer):
    if not viewer._segment_checks:
        pytest.skip("no masks found in the synthetic sample")
    viewer._set_all_segments(False)
    assert viewer._selected_segment_indices() == set()

    viewer._segment_checks[0].setChecked(True)
    assert viewer._selected_segment_indices() == {1}

    if len(viewer._segment_checks) > 1:
        viewer._segment_checks[1].setChecked(True)
        assert viewer._selected_segment_indices() == {1, 2}


def test_overlay_is_purple_at_half_opacity_over_the_greyscale(viewer):
    from echoviewer.ui import OVERLAY_ALPHA, OVERLAY_RGB

    viewer._set_all_segments(True)
    viewer._show_segmentation_check.setChecked(True)

    overlay = viewer._overlay_image
    assert overlay is not None and overlay.get_visible()
    rgba = overlay.get_array()
    painted = rgba[..., 3] > 0
    assert painted.any(), "nothing was painted"
    np.testing.assert_allclose(rgba[painted][0, :3], OVERLAY_RGB, atol=1e-6)
    assert rgba[..., 3].max() == pytest.approx(OVERLAY_ALPHA)
    # Transparent everywhere else, so the greyscale is not replaced.
    assert not np.asarray(rgba)[..., 3][~painted].any()
    assert viewer._image is not None
    assert overlay.get_zorder() > viewer._image.get_zorder()


def test_overlay_matches_the_displayed_plane_in_every_view(viewer):
    viewer._set_all_segments(True)
    for plane in range(3):
        viewer._view_combo.setCurrentIndex(plane)
        viewer._set_slice(viewer._slice_slider.maximum() // 2)
        expected = viewer._current_plane().image.shape
        assert viewer._overlay_image.get_array().shape[:2] == expected


def test_overlay_is_hidden_when_masks_do_not_fit_the_volume(viewer):
    """A mask must never be painted onto voxels it was not computed for."""
    from dataclasses import replace

    viewer._segmentation = replace(viewer._segmentation, source_shape=(999, 999, 999))
    assert viewer._label_plane() is None


def test_recompute_is_the_only_other_way_to_segment(viewer):
    baseline = viewer.segmentation_runs
    viewer._recompute_segmentation()
    assert viewer.segmentation_runs == baseline + 1
