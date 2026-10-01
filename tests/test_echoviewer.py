"""Tests for echo-time discovery, volume loading and the cached series."""

import numpy as np
import pytest

from echoviewer import (
    EchoSeries,
    get_available_echotimes,
    get_available_samples,
    load_volume_from_folder,
)

from dicomview.loader import DicomLoadError
from dicomview.volume import StackAxis, View


# --- discovery --------------------------------------------------------------


def test_samples_are_found_and_decoys_ignored(volumes_tree):
    assert get_available_samples(volumes_tree) == ["Pill1", "Syringes"]


def test_echotimes_sort_numerically_not_lexically(volumes_tree):
    """Lexical order would give 104.0, 16.0, 8.0 and scramble the decay curve."""
    echoes = get_available_echotimes(volumes_tree / "Syringes")
    assert [e.value for e in echoes] == [8.0, 16.0, 104.0]
    assert [e.path.name for e in echoes] == ["8.0", "16.0", "104.0"]


def test_echotime_value_comes_from_the_dicom_tag(volumes_tree):
    echoes = get_available_echotimes(volumes_tree / "Syringes")
    assert all(e.from_tag for e in echoes)
    assert echoes[0].label == "TE = 8 ms"


def test_prefixed_folder_names_are_supported(prefixed_tree):
    """The EchoTime_N convention resolves to the real TE from the header."""
    echoes = get_available_echotimes(prefixed_tree / "Sample")
    assert [e.path.name for e in echoes] == ["EchoTime_1", "EchoTime_2"]
    assert [e.value for e in echoes] == [12.0, 24.0]


def test_missing_root_raises(tmp_path):
    with pytest.raises(DicomLoadError):
        get_available_samples(tmp_path / "nowhere")


def test_directory_without_samples_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(DicomLoadError):
        get_available_samples(tmp_path / "empty")


# --- volume loading ---------------------------------------------------------


def test_volume_stacks_all_slices_in_order(volumes_tree):
    echoes = get_available_echotimes(volumes_tree / "Syringes")
    volume = load_volume_from_folder(echoes[0].path)
    assert volume.shape == (4, 6, 4)
    assert volume.stack_axis is StackAxis.SPATIAL
    # Values encode 100 * echo_index + slice_index; echo 0 gives 0,1,2,3.
    assert [float(volume.data[k].flat[0]) for k in range(4)] == pytest.approx([0, 1, 2, 3])


def test_each_echotime_loads_its_own_volume(volumes_tree):
    echoes = get_available_echotimes(volumes_tree / "Syringes")
    for echo_index, echo in enumerate(echoes):
        volume = load_volume_from_folder(echo.path)
        assert float(volume.data[0].flat[0]) == pytest.approx(100 * echo_index)


def test_slices_are_ordered_by_position_not_filename(volumes_tree, tmp_path):
    """Renaming the files must not change the stacking order.

    These volumes advance along X with a -X slice normal, so filename order
    and spatial order are easy to confuse; only the positions are authoritative.
    """
    import shutil

    source = get_available_echotimes(volumes_tree / "Syringes")[0].path
    shuffled = tmp_path / "shuffled"
    shuffled.mkdir()
    # Reverse the names so filename order is the opposite of spatial order.
    originals = sorted(source.glob("*.dcm"))
    for position, path in enumerate(reversed(originals)):
        shutil.copy(path, shuffled / f"zz_{position:03d}.dcm")

    volume = load_volume_from_folder(shuffled)
    assert [float(volume.data[k].flat[0]) for k in range(4)] == pytest.approx([0, 1, 2, 3])


def test_slice_spacing_comes_from_positions(volumes_tree):
    echoes = get_available_echotimes(volumes_tree / "Syringes")
    volume = load_volume_from_folder(echoes[0].path)
    assert volume.spacing == pytest.approx((0.5, 0.5, 0.5))


def test_all_three_planes_are_available(volumes_tree):
    """Plane extraction is inherited from dicomview and must still work."""
    echoes = get_available_echotimes(volumes_tree / "Syringes")
    volume = load_volume_from_folder(echoes[0].path)
    assert volume.n_slices(View.AXIAL) == 4
    assert volume.n_slices(View.CORONAL) == 6
    assert volume.n_slices(View.SAGITTAL) == 4
    for view in View:
        assert volume.extract(view, 0).image.ndim == 2


# --- EchoSeries -------------------------------------------------------------


def test_series_indexes_volumes_by_echo(volumes_tree):
    series = EchoSeries.from_sample(volumes_tree, "Syringes")
    assert len(series) == 3
    for echo_index in range(3):
        volume = series.volume(echo_index)
        assert float(volume.data[0].flat[0]) == pytest.approx(100 * echo_index)
    series.close()


def test_cache_returns_the_same_object(volumes_tree):
    series = EchoSeries.from_sample(volumes_tree, "Syringes")
    first = series.volume(1)
    assert series.is_cached(1)
    assert series.volume(1) is first
    series.close()


def test_cache_evicts_least_recently_used(volumes_tree):
    series = EchoSeries.from_sample(volumes_tree, "Syringes", cache_size=2)
    series.volume(0)
    series.volume(1)
    series.volume(2)
    assert not series.is_cached(0)
    assert series.cached_indices == [1, 2]
    series.close()


def test_out_of_range_echo_raises(volumes_tree):
    series = EchoSeries.from_sample(volumes_tree, "Syringes")
    with pytest.raises(IndexError):
        series.volume(99)
    series.close()


def test_index_of_nearest_echo_time(volumes_tree):
    series = EchoSeries.from_sample(volumes_tree, "Syringes")
    assert series.index_of_nearest(8.0) == 0
    assert series.index_of_nearest(15.0) == 1
    assert series.index_of_nearest(1000.0) == 2
    series.close()


def test_prefetch_warms_neighbours(volumes_tree):
    series = EchoSeries.from_sample(volumes_tree, "Syringes")
    series.volume(0)
    series.prefetch_around(0, radius=2)
    deadline = __import__("time").time() + 10
    while __import__("time").time() < deadline and not series.is_cached(1):
        __import__("time").sleep(0.02)
    assert series.is_cached(1), "prefetch thread did not warm the neighbouring echo"
    series.close()


def test_close_is_idempotent(volumes_tree):
    series = EchoSeries.from_sample(volumes_tree, "Syringes")
    series.volume(0)
    series.close()
    series.close()
    assert series.cached_indices == []


def test_context_manager_releases_cache(volumes_tree):
    with EchoSeries.from_sample(volumes_tree, "Syringes") as series:
        series.volume(0)
        assert series.cached_indices
    assert series.cached_indices == []


def test_empty_sample_rejected():
    with pytest.raises(ValueError):
        EchoSeries("Nothing", [])


# --- incomplete-export detection -------------------------------------------


def test_inconsistent_echoes_flags_the_short_folder(ragged_tree):
    series = EchoSeries.from_sample(ragged_tree, "Syringes")
    assert series.inconsistent_echoes() == [(1, 2)]  # echo index 1 has 2 slices
    series.close()


def test_uniform_sample_reports_nothing(volumes_tree):
    series = EchoSeries.from_sample(volumes_tree, "Syringes")
    assert series.inconsistent_echoes() == []
    series.close()


def test_short_echo_folder_still_loads(ragged_tree):
    """An incomplete export must load as a smaller volume, not fail."""
    echoes = get_available_echotimes(ragged_tree / "Syringes")
    volume = load_volume_from_folder(echoes[1].path)
    assert volume.shape[0] == 2


# --- UI: a failing echo must not take the process down ----------------------


def test_show_echo_reports_errors_instead_of_propagating(qt_app, volumes_tree, monkeypatch):
    """An exception escaping this slot would make PyQt5 abort the process.

    That is what makes the window vanish with no dialog, so the handler has to
    swallow everything and report it.
    """
    from echoviewer import ui as ui_module

    window = ui_module.EchoViewerWindow(volumes_tree, sample="Syringes")
    shown: list[str] = []
    monkeypatch.setattr(
        ui_module.QMessageBox, "critical", staticmethod(lambda *a, **k: shown.append(a[2]))
    )

    def explode(self, index):
        raise ValueError("simulated bad volume")

    monkeypatch.setattr(type(window._series), "volume", explode)
    window._show_echo(1)  # must not raise

    assert shown, "the failure was not reported to the user"
    assert "simulated bad volume" in shown[0]
    window.close()


def test_dragging_the_echo_slider_defers_uncached_loads(qt_app, volumes_tree):
    """A fast drag must not trigger one synchronous load per value passed."""
    from echoviewer import ui as ui_module

    window = ui_module.EchoViewerWindow(volumes_tree, sample="Syringes")
    window._series.clear()  # nothing cached, so every step would load

    loads: list[int] = []
    original = type(window._series).volume
    type(window._series).volume = lambda self, i: (loads.append(i), original(self, i))[1]
    try:
        for index in range(len(window._series)):
            window._echo_slider.setValue(index)
        assert window._pending_echo == len(window._series) - 1
        assert not loads, f"drag loaded {len(loads)} volumes synchronously"
    finally:
        type(window._series).volume = original
        window.close()


# --- the window must survive garbage collection ------------------------------


def test_window_survives_garbage_collection(qt_app, volumes_tree):
    """The regression behind "it just closed while I was using it".

    PyQt keeps no Python reference to a parentless top-level window. The window
    and its children reference each other through signal connections, so it is
    freed by the *cyclic* collector rather than by refcounting - at an
    unpredictable moment, which loading volumes makes arrive sooner. If the
    only reference is a discarded return value, the window is destroyed
    mid-session and the app exits cleanly with no traceback.
    """
    import gc

    try:  # sip ships inside the PyQt5 package on some builds, standalone on others
        from PyQt5 import sip
    except ImportError:  # pragma: no cover - depends on the PyQt5 build
        import sip

    from echoviewer import ui as ui_module

    window = ui_module.create_viewer_with_echotime_slider(
        volumes_tree, sample="Syringes", show=False
    )
    # Drop the caller's reference, exactly as the old run() did.
    del window
    gc.collect()
    gc.collect()

    assert ui_module._OPEN_WINDOWS, "no strong reference kept; the window can be collected"
    survivor = ui_module._OPEN_WINDOWS[-1]
    assert not sip.isdeleted(survivor), "the Qt window was destroyed by garbage collection"
    # Still functional after collection.
    survivor._echo_slider.setValue(1)
    survivor.close()


def test_closing_a_window_releases_the_reference(qt_app, volumes_tree):
    """The registry must not leak windows across a long session."""
    from echoviewer import ui as ui_module

    from PyQt5.QtCore import QEvent

    before = len(ui_module._OPEN_WINDOWS)
    window = ui_module.create_viewer_with_echotime_slider(
        volumes_tree, sample="Syringes", show=False
    )
    assert len(ui_module._OPEN_WINDOWS) == before + 1

    window.close()
    window.deleteLater()
    # deleteLater posts a DeferredDelete event, which processEvents() does not
    # dispatch on its own; the destroyed signal fires only once it is delivered.
    qt_app.processEvents()
    qt_app.sendPostedEvents(None, QEvent.DeferredDelete)
    qt_app.processEvents()

    assert len(ui_module._OPEN_WINDOWS) == before


# --- integration with the real dataset --------------------------------------


def test_real_dataset_samples_and_echoes(real_volumes):
    samples = get_available_samples(real_volumes)
    assert {"Syringes", "3D", "Pill1", "Pill2"} <= set(samples)

    echoes = get_available_echotimes(real_volumes / "Syringes")
    assert len(echoes) == 26
    assert [e.value for e in echoes] == pytest.approx(list(range(8, 209, 8)))


def test_real_volume_loads_with_expected_geometry(real_volumes):
    series = EchoSeries.from_sample(real_volumes, "Syringes")
    volume = series.volume(0)
    assert volume.shape == (128, 330, 128)
    assert volume.spacing == pytest.approx((0.32, 0.32, 0.32))
    assert volume.stack_axis is StackAxis.SPATIAL
    series.close()


def test_real_signal_decays_with_echo_time(real_volumes):
    """The physics check: mean signal must fall as TE rises."""
    series = EchoSeries.from_sample(real_volumes, "Syringes", cache_size=3)
    means = [float(np.mean(series.volume(i).data)) for i in (0, 12, 25)]
    assert means[0] > means[1] > means[2], f"signal did not decay: {means}"
    series.close()
