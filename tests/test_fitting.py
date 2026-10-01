"""Tests for the voxel-wise decay fitting.

What these protect:

* both models recovering known parameters from synthetic decays;
* every voxel fitted independently — no averaging before the fit;
* spatial shape preserved, background left at zero, coordinates aligned;
* segmentation ``.npz`` zeros never fitted and never cropped away;
* failures degrading to zeros and a cleared success flag, never an exception;
* segmentation folder names, and their concentration labels, carried through.
"""

import json

import numpy as np
import pytest

from echoviewer.fitting import (
    MAP_NAMES,
    MODEL_NO_BASELINE,
    MODEL_WITH_BASELINE,
    build_maps,
    fit_one_voxel,
    fit_signals,
    model_no_baseline,
    model_with_baseline,
    summarise,
    write_fit_outputs,
)
from echoviewer.relaxometry import (
    find_echo_folders,
    find_segmentations,
    parse_segmentation_name,
)

TE = np.arange(8.0, 209.0, 8.0)


# --- the models recover what they should ------------------------------------


def test_model_a_recovers_its_parameters():
    signal = model_no_baseline(TE, 2000.0, 0.025)
    result = fit_one_voxel(TE, signal)
    assert result["I0_noB"] == pytest.approx(2000.0, rel=1e-3)
    assert result["rs_noB"] == pytest.approx(0.025, rel=1e-3)
    assert result["R2_noB"] > 0.999


def test_model_b_recovers_a_real_baseline():
    """B is free, not assumed zero."""
    signal = model_with_baseline(TE, 1500.0, 0.03, 120.0)
    result = fit_one_voxel(TE, signal)
    assert result["I0_B"] == pytest.approx(1500.0, rel=1e-2)
    assert result["rs_B"] == pytest.approx(0.03, rel=1e-2)
    assert result["B"] == pytest.approx(120.0, rel=1e-2)
    assert result["R2_B"] > 0.999


def test_both_models_are_kept(sample_result=None):
    """Neither fit is discarded, whichever the criterion prefers."""
    result = fit_one_voxel(TE, model_with_baseline(TE, 1500.0, 0.03, 120.0))
    for key in ("I0_noB", "rs_noB", "R2_noB", "RMSE_noB",
                "I0_B", "rs_B", "B", "R2_B", "RMSE_B", "delta_R2"):
        assert key in result
    assert result["success_noB"] == 1 and result["success_B"] == 1


def test_baseline_data_prefers_model_b():
    result = fit_one_voxel(TE, model_with_baseline(TE, 1500.0, 0.03, 200.0))
    assert result["model_selected"] == MODEL_WITH_BASELINE
    assert result["delta_R2"] > 0


def test_clean_exponential_prefers_model_a():
    """AIC, not R2: the extra parameter has to earn its place."""
    rng = np.random.default_rng(0)
    signal = model_no_baseline(TE, 2000.0, 0.025) + rng.normal(0, 2.0, TE.size)
    result = fit_one_voxel(TE, signal)
    assert result["R2_B"] >= result["R2_noB"] - 1e-9  # R2 can only improve
    assert result["model_selected"] == MODEL_NO_BASELINE


def test_noise_floor_biases_model_a_low():
    """Why both models are kept: ignoring a floor distorts the rate."""
    signal = model_with_baseline(TE, 2000.0, 0.04, 150.0)
    result = fit_one_voxel(TE, signal)
    assert result["rs_B"] == pytest.approx(0.04, rel=1e-2)
    assert result["rs_noB"] < result["rs_B"]


# --- robustness -------------------------------------------------------------


def test_flat_signal_is_not_fitted():
    result = fit_one_voxel(TE, np.full(TE.size, 500.0))
    assert result["fit_success"] == 0
    assert result["model_selected"] == 0


def test_all_zero_signal_is_not_fitted():
    result = fit_one_voxel(TE, np.zeros(TE.size))
    assert result["fit_success"] == 0


def test_nan_and_inf_are_dropped_not_fatal():
    signal = model_no_baseline(TE, 1000.0, 0.02).copy()
    signal[3] = np.nan
    signal[7] = np.inf
    result = fit_one_voxel(TE, signal)
    assert result["n_points"] == TE.size - 2
    assert result["success_noB"] == 1
    assert result["I0_noB"] == pytest.approx(1000.0, rel=1e-2)


def test_too_few_points_is_reported_not_raised():
    result = fit_one_voxel(np.array([8.0, 16.0]), np.array([100.0, 50.0]))
    assert result["fit_success"] == 0
    assert result["n_points"] == 2


def test_failure_leaves_zeros_everywhere():
    result = fit_one_voxel(TE, np.full(TE.size, np.nan))
    for key in ("I0_noB", "rs_noB", "I0_B", "rs_B", "B", "R2_noB", "R2_B"):
        assert result[key] == 0.0


def test_parameters_stay_in_bounds():
    rng = np.random.default_rng(1)
    signal = model_no_baseline(TE, 3000.0, 0.05) + rng.normal(0, 200, TE.size)
    result = fit_one_voxel(TE, np.clip(signal, 0, None))
    assert result["rs_noB"] >= 0 and result["rs_B"] >= 0
    assert result["I0_noB"] >= 0 and result["B"] >= 0


# --- many voxels ------------------------------------------------------------


def test_each_voxel_is_fitted_independently():
    """Different voxels must give different answers - nothing is averaged."""
    rates = np.array([0.01, 0.03, 0.06])
    signals = np.stack([model_no_baseline(TE, 1000.0, r) for r in rates])
    results = fit_signals(TE, signals, workers=1)
    np.testing.assert_allclose(results["rs_noB"], rates, rtol=1e-3)
    # And the mean of the three curves is not what any voxel reports.
    mean_rate = fit_one_voxel(TE, signals.mean(axis=0))["rs_noB"]
    assert not np.allclose(results["rs_noB"], mean_rate)


def test_empty_input_is_handled():
    results = fit_signals(TE, np.zeros((0, TE.size)), workers=1)
    assert results["rs_noB"].size == 0


# --- maps -------------------------------------------------------------------


@pytest.fixture
def small_fit():
    """Three fitted voxels scattered in a 4x5x6 volume."""
    shape = (4, 5, 6)
    coords = (np.array([0, 2, 3]), np.array([1, 2, 4]), np.array([5, 0, 3]))
    rates = np.array([0.01, 0.03, 0.06])
    signals = np.stack([model_no_baseline(TE, 1000.0, r) for r in rates])
    results = fit_signals(TE, signals, workers=1)
    return shape, coords, results


def test_maps_keep_the_full_spatial_shape(small_fit):
    shape, coords, results = small_fit
    maps = build_maps(results, coords, shape)
    assert set(maps) == set(MAP_NAMES)
    for name, volume in maps.items():
        assert volume.shape == shape, f"{name} was reshaped"


def test_maps_are_zero_outside_the_fitted_voxels(small_fit):
    shape, coords, results = small_fit
    maps = build_maps(results, coords, shape)
    outside = np.ones(shape, dtype=bool)
    outside[coords] = False
    for name, volume in maps.items():
        assert not volume[outside].any(), f"{name} is non-zero outside"


def test_map_values_land_on_the_right_voxels(small_fit):
    shape, coords, results = small_fit
    maps = build_maps(results, coords, shape)
    for position in range(coords[0].size):
        index = (coords[0][position], coords[1][position], coords[2][position])
        assert maps["rs_map_noB"][index] == pytest.approx(
            results["rs_noB"][position], rel=1e-5
        )


def test_outputs_are_written_and_reload(tmp_path, small_fit):
    import pandas as pd

    shape, coords, results = small_fit
    maps = build_maps(results, coords, shape)
    metadata = {"shape": list(shape), "echo_times": TE.tolist(), "source": "test"}
    write_fit_outputs(tmp_path, maps, coords, results, metadata)

    for name in MAP_NAMES:
        assert np.load(tmp_path / f"{name}.npy").shape == shape

    frame = pd.read_csv(tmp_path / "voxel_fit_data.csv")
    assert len(frame) == coords[0].size
    for column in ("x", "y", "z", "I0_noB", "rs_noB", "I0_B", "rs_B", "B",
                   "R2_noB", "R2_B", "RMSE_noB", "RMSE_B", "delta_R2",
                   "model_selected", "fit_success"):
        assert column in frame.columns

    assert json.loads((tmp_path / "metadata.json").read_text())["shape"] == list(shape)


def test_csv_coordinates_match_the_maps(tmp_path, small_fit):
    import pandas as pd

    shape, coords, results = small_fit
    maps = build_maps(results, coords, shape)
    write_fit_outputs(tmp_path, maps, coords, results, {"shape": list(shape)})

    frame = pd.read_csv(tmp_path / "voxel_fit_data.csv")
    stored = np.load(tmp_path / "rs_map_noB.npy")
    for _, row in frame.iterrows():
        assert stored[int(row.z), int(row.y), int(row.x)] == pytest.approx(
            row.rs_noB, rel=1e-4
        )


def test_summary_counts_models(small_fit):
    shape, coords, results = small_fit
    report = summarise("t", "test", tmp_path_placeholder := __import__("pathlib").Path("."),
                       shape, TE.tolist(), results)
    assert report.n_voxels == 3
    assert sum(report.model_counts.values()) == 3
    assert "rs_noB" in report.stats
    assert tmp_path_placeholder  # keep the linter quiet


# --- segmentation naming and discovery --------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Segmentation_6_0.2", (6, "0.2")),
        ("Segmentation_4_water", (4, "water")),
        ("Segmentation_2_tissue", (2, "tissue")),
        ("Segmentation_1_0", (1, "0")),
        ("Segmentation_3", (3, "")),
    ],
)
def test_segmentation_labels_are_parsed_not_lost(name, expected):
    assert parse_segmentation_name(name) == expected


def test_non_segmentation_folders_are_ignored():
    assert parse_segmentation_name("images_segmentations") is None
    assert parse_segmentation_name("metadata.json") is None


def test_echo_folders_sort_numerically(tmp_path):
    """Lexical order would put 104.0 before 16.0 and scramble every curve."""
    for name in ("8.0", "16.0", "104.0", "24.0"):
        folder = tmp_path / name
        folder.mkdir()
        np.savez_compressed(folder / "volume.npz", volume=np.zeros((2, 2, 2)))
    assert [value for value, _ in find_echo_folders(tmp_path)] == [8.0, 16.0, 24.0, 104.0]


def test_segmentation_discovery_preserves_names(tmp_path):
    volumes = tmp_path / "volumes"
    for name in ("Segmentation_2_water", "Segmentation_1_0.2", "images_segmentations"):
        (volumes / name).mkdir(parents=True)
    found = [folder.name for folder in find_segmentations(tmp_path)]
    assert found == ["Segmentation_1_0.2", "Segmentation_2_water"]


# --- end to end on a synthetic segmentation ---------------------------------


def test_segmentation_zeros_are_never_fitted(tmp_path):
    """The .npz background is outside the segmentation and must stay untouched."""
    from echoviewer.relaxometry import fit_segmentation

    shape = (3, 6, 6)
    inside = np.zeros(shape, dtype=bool)
    inside[1, 2:4, 2:4] = True

    segmentation_dir = tmp_path / "volumes" / "Segmentation_1_0.2"
    for echo in (8.0, 16.0, 24.0, 32.0, 40.0):
        folder = segmentation_dir / f"{echo}"
        folder.mkdir(parents=True)
        volume = np.zeros(shape, dtype=np.float32)
        volume[inside] = 1000.0 * np.exp(-0.02 * echo)
        np.savez_compressed(folder / "volume.npz", volume=volume)

    report = fit_segmentation(
        segmentation_dir, tmp_path / "out", "TestSample", workers=1
    )

    assert report.shape == shape          # not cropped
    assert report.n_voxels == int(inside.sum())

    rate_map = np.load(tmp_path / "out" / "rs_map_noB.npy")
    assert rate_map.shape == shape
    assert not rate_map[~inside].any(), "background outside the segmentation was fitted"
    np.testing.assert_allclose(rate_map[inside], 0.02, rtol=1e-2)

    metadata = json.loads((tmp_path / "out" / "metadata.json").read_text())
    assert metadata["segmentation_name"] == "Segmentation_1_0.2"
    assert metadata["segmentation_label"] == "0.2"
    assert metadata["source"] == "segmentation"
    assert metadata["echo_times"] == [8.0, 16.0, 24.0, 32.0, 40.0]
    assert metadata["shape"] == list(shape)
