"""Qt UI adding an echo-time axis to the original DICOM viewer.

:class:`EchoViewerWindow` subclasses :class:`dicomview.ui.ViewerWindow` rather
than reimplementing it, so plane switching, the slice slider, window/level,
zoom/pan and SVG export all come from the original viewer unchanged. This
module adds exactly one new thing: a second slider that moves through echo
times, swapping the whole volume underneath the existing views.

Two behaviours are deliberate and worth knowing about:

* **The window is held fixed when the echo time changes.** Auto-windowing each
  echo separately would rescale every image to its own range and visually
  erase the T2 decay that the echo axis exists to show. Signal really does
  drop with TE, and the display should show that. There is a checkbox to opt
  into per-echo auto-windowing when you just want to see structure.
* **The slice index and plane are preserved** across an echo change, so you
  stay on the same anatomy while stepping through TE.

Coupling: this subclass uses the parent's internal attributes (``_volume``,
``_view``, ``_index``, ``_image``, ``_axes``, ``_canvas``) and its private
build hooks. That is the price of extending the original without editing it;
the names are listed here so a future change to the parent has one obvious
place to check.
"""

from __future__ import annotations

import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

from . import _bootstrap  # noqa: F401  (puts dicomview on sys.path)
from .dataset import get_available_samples
from .segmentation import AUTO_PRESET, PRESETS, Segmentation, segment_volume
from .series import DEFAULT_CACHE_SIZE, EchoSeries

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

#: Overlay colour for segmentation masks, and its opacity. Purple reads clearly
#: against a greyscale MR image at half opacity, where a red or green would
#: compete with the tissue itself.
#:
#: Imported rather than defined here so the saved preview images and the screen
#: share one definition and cannot drift apart.
from .preview import OVERLAY_ALPHA, OVERLAY_RGB  # noqa: E402,F401

#: Wait this long after the last echo-slider movement before loading. Dragging
#: across the slider would otherwise trigger a synchronous ~0.5 s load for every
#: value it passes through, freezing the UI for many seconds.
ECHO_DEBOUNCE_MS = 150

#: Unhandled exceptions are appended here as well as shown in a dialog.
ERROR_LOG = Path(__file__).resolve().parent.parent / "echoviewer_errors.log"


def install_exception_logging(log_path: Path = ERROR_LOG) -> None:
    """Stop an unhandled exception from killing the application silently.

    PyQt5 responds to an exception escaping a slot by calling ``qFatal()``,
    which aborts the process immediately - the window just vanishes, with the
    traceback going to a terminal the user may not be watching. Installing any
    custom ``sys.excepthook`` suppresses that abort, so this one records the
    traceback and tells the user where to find it, leaving the app running.
    """

    def hook(exc_type, exc, tb) -> None:
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        try:
            with open(log_path, "a", encoding="utf-8") as handle:
                handle.write(f"\n=== {datetime.now().isoformat(timespec='seconds')} ===\n{text}")
        except OSError:
            pass  # Logging must never itself take the app down.
        sys.stderr.write(text)
        if QApplication.instance() is not None:
            QMessageBox.critical(
                None,
                "Unexpected error",
                f"{exc_type.__name__}: {exc}\n\n"
                f"The viewer is still running. Details were written to:\n{log_path}",
            )

    sys.excepthook = hook

from matplotlib.patheffects import withStroke

from dicomview.loader import DicomLoadError
from dicomview.ui import ViewerWindow
from dicomview.volume import Volume
from dicomview.windowing import default_window


class EchoViewerWindow(ViewerWindow):
    """The original viewer plus a sample selector and an echo-time slider.

    Args:
        volumes_root: The ``Volumes`` directory holding the samples.
        sample: Sample to open first. Defaults to the first one found.
        cache_size: Volumes held in memory per sample.
    """

    def __init__(
        self,
        volumes_root: str | Path,
        sample: str | None = None,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ):
        # The parent's __init__ calls the build hooks overridden below, and
        # those run before any of this instance's own attributes exist, so
        # they all guard with getattr().
        super().__init__()

        self._root = Path(volumes_root)
        self._series: EchoSeries | None = None
        self._echo_index = 0
        self._cache_size = cache_size

        # Segmentation state. The masks are tied to a sample, not to an echo
        # time, so nothing here is touched when the echo slider moves.
        self._segmentation: Segmentation | None = None
        self._label_volume: Volume | None = None
        self._segmentation_runs = 0
        self._overlay_image = None
        self._overlay_texts: list = []

        try:
            self._samples = get_available_samples(self._root)
        except DicomLoadError as exc:
            QMessageBox.critical(self, "Could not read Volumes folder", str(exc))
            self._samples = []
            return

        self._series_combo.blockSignals(True)
        self._series_combo.clear()
        self._series_combo.addItems(self._samples)
        self._series_combo.blockSignals(False)

        first = sample if sample in self._samples else self._samples[0]
        if sample is not None and sample not in self._samples:
            QMessageBox.warning(
                self,
                "Sample not found",
                f"No sample named {sample!r}. Available: {', '.join(self._samples)}.\n"
                f"Opening {first!r} instead.",
            )
        # Setting the combo emits currentIndexChanged, which would load and
        # segment the sample a first time before the explicit call below did it
        # again. Block it and load exactly once.
        self._series_combo.blockSignals(True)
        self._series_combo.setCurrentText(first)
        self._series_combo.blockSignals(False)
        self._load_sample(first)

    # ------------------------------------------------------------------
    # Build hooks (called from the parent's __init__)
    # ------------------------------------------------------------------

    def _build_series_box(self) -> QWidget:
        """Reuse the parent's series box as a sample selector, plus echo controls."""
        sample_box = super()._build_series_box()
        sample_box.setTitle("Sample")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(sample_box)
        layout.addWidget(self._build_echo_box())
        layout.addWidget(self._build_segmentation_box())
        return container

    def _build_segmentation_box(self) -> QGroupBox:
        """Controls for the cached 3D masks.

        None of these widgets ever triggers segmentation. The only two things
        that do are opening a sample and pressing Recompute; everything here
        just changes what is drawn.
        """
        box = QGroupBox("Segmentation")
        layout = QVBoxLayout(box)

        self._show_segmentation_check = QCheckBox("Show segmentations")
        self._show_segmentation_check.setChecked(True)
        self._show_segmentation_check.setToolTip(
            "Master switch. Turning it off hides every overlay but keeps the "
            "individual selections below."
        )
        self._show_segmentation_check.stateChanged.connect(lambda _: self._refresh())
        layout.addWidget(self._show_segmentation_check)

        # Individual masks live in a scroll area: a sample can produce more
        # regions than fit in the panel.
        self._segment_checks: list[QCheckBox] = []
        self._segment_container = QWidget()
        self._segment_layout = QVBoxLayout(self._segment_container)
        self._segment_layout.setContentsMargins(4, 2, 4, 2)
        self._segment_layout.setSpacing(2)
        self._segment_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(self._segment_container)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(110)
        scroll.setMaximumHeight(190)
        layout.addWidget(scroll)

        buttons = QHBoxLayout()
        all_button = QPushButton("All")
        all_button.clicked.connect(lambda: self._set_all_segments(True))
        none_button = QPushButton("None")
        none_button.clicked.connect(lambda: self._set_all_segments(False))
        buttons.addWidget(all_button)
        buttons.addWidget(none_button)
        row = QWidget()
        row.setLayout(buttons)
        layout.addWidget(row)

        preset_row = QHBoxLayout()
        self._preset_combo = QComboBox()
        self._preset_combo.addItem(AUTO_PRESET)
        self._preset_combo.addItems(list(PRESETS))
        self._preset_combo.setToolTip(
            "Changing this does not recompute on its own - press Recompute."
        )
        recompute = QPushButton("Recompute")
        recompute.setToolTip("Run the segmentation again on the first echo time.")
        recompute.clicked.connect(self._recompute_segmentation)
        preset_row.addWidget(self._preset_combo, stretch=1)
        preset_row.addWidget(recompute)
        preset_widget = QWidget()
        preset_widget.setLayout(preset_row)
        layout.addWidget(preset_widget)

        self._save_button = QPushButton("Save segmentations to disk...")
        self._save_button.setToolTip(
            "Write the masks already computed for this sample as sparse tables "
            "and dense volumes. Does not re-run the segmentation."
        )
        self._save_button.clicked.connect(self._export_segmentations)
        layout.addWidget(self._save_button)

        self._segmentation_label = QLabel("-")
        self._segmentation_label.setWordWrap(True)
        self._segmentation_label.setStyleSheet("color:#555; font-size:11px;")
        layout.addWidget(self._segmentation_label)
        return box

    def _export_segmentations(self) -> None:
        """Write this sample's cached masks to disk, sparse and dense.

        Reuses ``self._segmentation`` untouched - the export is a pure write of
        what is already on screen, so nothing is recomputed and the saved masks
        cannot differ from the displayed ones.
        """
        series = getattr(self, "_series", None)
        if series is None or self._segmentation is None or len(self._segmentation) == 0:
            QMessageBox.information(
                self, "Nothing to save", "This sample has no segmentations."
            )
            return

        directory = QFileDialog.getExistingDirectory(
            self, "Choose the Segmentations folder", str(self._root.parent)
        )
        if not directory:
            return

        runs_before = self._segmentation_runs
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        try:
            from .export import export_sample

            report = export_sample(
                series,
                self._segmentation,
                directory,
                progress=lambda done, total, message: (
                    self.statusBar().showMessage(f"Saving {message} ({done}/{total})"),
                    QApplication.processEvents(),
                ),
            )
        except Exception as exc:  # noqa: BLE001 - must not abort the process
            traceback.print_exc()
            QMessageBox.critical(self, "Save failed", f"{type(exc).__name__}: {exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        # The export must not have touched the segmentation.
        assert self._segmentation_runs == runs_before
        message = report.summary + f"\n\nWritten to:\n{report.root}"
        if report.skipped:
            message += "\n\nSkipped:\n" + "\n".join(
                f"TE {value:g} ms - {reason}" for value, reason in report.skipped
            )
        QMessageBox.information(self, "Segmentations saved", message)
        self.statusBar().showMessage(report.summary, 8000)

    def _build_echo_box(self) -> QGroupBox:
        """The new echo-time controls."""
        box = QGroupBox("Echo time")
        layout = QFormLayout(box)

        self._echo_slider = QSlider(Qt.Horizontal)
        self._echo_slider.setRange(0, 0)
        self._echo_slider.setPageStep(1)
        self._echo_slider.valueChanged.connect(self._echo_changed)

        # Dragging emits valueChanged for every value passed over. Loading each
        # one synchronously would freeze the UI, so uncached echoes wait until
        # the slider settles; cached ones still switch instantly.
        self._pending_echo: int | None = None
        self._echo_timer = QTimer(self)
        self._echo_timer.setSingleShot(True)
        self._echo_timer.setInterval(ECHO_DEBOUNCE_MS)
        self._echo_timer.timeout.connect(self._apply_pending_echo)

        buttons = QHBoxLayout()
        previous = QPushButton("<")
        previous.setFixedWidth(30)
        previous.setToolTip("Previous echo time")
        previous.clicked.connect(lambda: self._set_echo(self._echo_index - 1))
        following = QPushButton(">")
        following.setFixedWidth(30)
        following.setToolTip("Next echo time")
        following.clicked.connect(lambda: self._set_echo(self._echo_index + 1))
        buttons.addWidget(previous)
        buttons.addWidget(self._echo_slider, stretch=1)
        buttons.addWidget(following)
        row = QWidget()
        row.setLayout(buttons)
        layout.addRow("TE", row)

        self._echo_label = QLabel("-")
        self._echo_label.setStyleSheet("color:#555; font-size:11px;")
        layout.addRow("", self._echo_label)

        self._auto_window_check = QCheckBox("Auto-window each echo")
        self._auto_window_check.setToolTip(
            "Off (default): one window for all echo times, so the T2 signal "
            "decay stays visible.\n"
            "On: rewindow every echo to its own range, which shows structure "
            "but hides the decay."
        )
        self._auto_window_check.setChecked(False)
        self._auto_window_check.stateChanged.connect(self._auto_window_toggled)
        layout.addRow("", self._auto_window_check)
        return box

    def _set_controls_enabled(self, enabled: bool) -> None:
        """Extend the parent's enable/disable to cover the echo controls."""
        super()._set_controls_enabled(enabled)
        for name in ("_echo_slider", "_auto_window_check"):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Sample handling
    # ------------------------------------------------------------------

    def _series_changed(self, index: int) -> None:
        """The parent's series combo now selects a sample."""
        samples = getattr(self, "_samples", [])
        if 0 <= index < len(samples):
            self._load_sample(samples[index])

    def _load_sample(self, name: str) -> None:
        """Open a sample: build its echo series and show the first echo."""
        previous = getattr(self, "_series", None)
        if previous is not None:
            previous.close()

        try:
            series = EchoSeries.from_sample(self._root, name, cache_size=self._cache_size)
            # Index 0 is the first echo time: echoes are sorted ascending by TE,
            # and this is the volume the masks are derived from.
            volume = series.volume(0)
        except (DicomLoadError, ValueError, IndexError) as exc:
            QMessageBox.critical(self, f"Could not load sample {name}", str(exc))
            self._series = None
            return

        self._series = series
        self._echo_index = 0
        self._pending_echo = None
        self._echo_timer.stop()
        # Computed before the first draw so the banner is right from the start.
        self._sample_warning = self._incomplete_echo_warning(series)

        self._echo_slider.blockSignals(True)
        self._echo_slider.setRange(0, len(series) - 1)
        self._echo_slider.setValue(0)
        self._echo_slider.blockSignals(False)
        self._echo_slider.setEnabled(len(series) > 1)

        # Segment the first echo time, once, before the first draw. Every other
        # echo time reuses these masks at the same voxel coordinates.
        self._overlay_image = None
        self._run_segmentation(volume, self._preset_combo.currentText())

        # Reset plane, slice and window for the new sample.
        self._load_volume(volume)
        self._set_controls_enabled(True)
        self._update_echo_label()
        series.prefetch_around(0)
        self.statusBar().showMessage(
            f"{name}: {len(series)} echo times, "
            f"TE {series.echotime(0).value:g}-{series.echotime(-1).value:g} ms",
            6000,
        )

    @staticmethod
    def _incomplete_echo_warning(series: EchoSeries) -> str:
        """Message naming echo folders with fewer slices than the rest, if any."""
        try:
            odd = series.inconsistent_echoes()
        except OSError:
            return ""
        if not odd:
            return ""
        details = ", ".join(
            f"{series.echotime(index).label} ({count} slices)" for index, count in odd
        )
        return (
            f"<b>{series.name}</b>: incomplete echo folder(s) - {details}. "
            "The rest of this sample has more slices, so stepping onto these "
            "changes the geometry and resets the slice slider."
        )

    def _update_banner(self) -> None:
        """Keep the sample-level warning visible alongside the parent's.

        The parent hides the banner whenever the current volume has no
        warnings of its own, which would erase the incomplete-echo notice on
        the first echo change.
        """
        super()._update_banner()
        sample_warning = getattr(self, "_sample_warning", "")
        if not sample_warning:
            return
        existing = self._banner.text() if self._banner.isVisibleTo(self) else ""
        self._banner.setText(
            f"{existing}<br>{sample_warning}" if existing else sample_warning
        )
        self._banner.show()

    # ------------------------------------------------------------------
    # Echo handling
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Segmentation: computed once per sample, then reused
    # ------------------------------------------------------------------

    @property
    def segmentation_runs(self) -> int:
        """How many times the segmentation algorithm has been invoked.

        Exposed so tests can assert that navigating echo times, slices, planes
        and checkboxes never re-runs it.
        """
        return self._segmentation_runs

    def _run_segmentation(self, volume: Volume, preset: str) -> None:
        """Segment ``volume`` and cache the result. The only place this happens.

        Called when a sample is opened - with the *first* echo time's volume -
        and when the user presses Recompute. Nothing else calls it, which is
        what keeps the masks echo-time invariant.
        """
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        self.statusBar().showMessage("Segmenting the first echo time...")
        QApplication.processEvents()
        try:
            self._segmentation = segment_volume(volume, preset=preset)
            self._segmentation_runs += 1
        except Exception as exc:  # noqa: BLE001 - must not abort the process
            traceback.print_exc()
            self._segmentation = None
            QMessageBox.critical(
                self, "Segmentation failed", f"{type(exc).__name__}: {exc}"
            )
        finally:
            QApplication.restoreOverrideCursor()

        # The label map is wrapped as a Volume once, so that every cross-section
        # is taken by the same code that slices the greyscale data.
        self._label_volume = (
            self._segmentation.as_volume() if self._segmentation is not None else None
        )
        self._rebuild_segment_checkboxes()

    def _recompute_segmentation(self) -> None:
        """Explicit user request: re-run on the first echo with the chosen preset."""
        series = getattr(self, "_series", None)
        if series is None:
            return
        try:
            first_volume = series.volume(0)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not load first echo time", str(exc))
            return
        self._run_segmentation(first_volume, self._preset_combo.currentText())
        self._refresh()

    def _rebuild_segment_checkboxes(self) -> None:
        """Replace the per-mask checkboxes to match the current segmentation."""
        for check in self._segment_checks:
            self._segment_layout.removeWidget(check)
            check.deleteLater()
        self._segment_checks = []

        segmentation = self._segmentation
        if segmentation is None or len(segmentation) == 0:
            self._segmentation_label.setText(
                "No segments found." if segmentation is not None else "Not segmented."
            )
            return

        for segment in segmentation.segments:
            check = QCheckBox(f"{segment.name}  ({segment.voxels:,} vox)")
            check.setChecked(True)
            check.stateChanged.connect(lambda _: self._refresh())
            # Insert before the trailing stretch so the list stays top-aligned.
            self._segment_layout.insertWidget(self._segment_layout.count() - 1, check)
            self._segment_checks.append(check)

        note = f"{len(segmentation)} segments - preset: {segmentation.preset}"
        if segmentation.warnings:
            note += "<br>" + "<br>".join(segmentation.warnings)
        self._segmentation_label.setText(note)

    def _set_all_segments(self, checked: bool) -> None:
        """Tick or clear every individual mask, then redraw once."""
        for check in self._segment_checks:
            check.blockSignals(True)
            check.setChecked(checked)
            check.blockSignals(False)
        self._refresh()

    def _selected_segment_indices(self) -> set[int]:
        """Label values the user currently wants drawn."""
        if not self._show_segmentation_check.isChecked():
            return set()
        return {
            position + 1
            for position, check in enumerate(self._segment_checks)
            if check.isChecked()
        }

    # ------------------------------------------------------------------
    # Echo handling
    # ------------------------------------------------------------------

    def _echo_changed(self, index: int) -> None:
        """Slider callback: switch now if cached, otherwise once the drag settles."""
        series = getattr(self, "_series", None)
        if series is None:
            return

        self._pending_echo = index
        self._preview_echo_label(index)

        if series.is_cached(index):
            self._echo_timer.stop()
            self._apply_pending_echo()
        else:
            # Let the drag continue; the volume loads when the user pauses.
            self._echo_timer.start()

    def _apply_pending_echo(self) -> None:
        """Load and display whichever echo the slider last settled on."""
        index = self._pending_echo
        if index is not None:
            self._pending_echo = None
            self._show_echo(index)

    def _set_echo(self, index: int) -> None:
        """Move to an echo index programmatically, via the normal slider path."""
        series = getattr(self, "_series", None)
        if series is None:
            return
        index = min(max(index, 0), len(series) - 1)
        if index == self._echo_index:
            return
        self._echo_slider.setValue(index)  # emits valueChanged -> _echo_changed

    def _show_echo(self, index: int) -> None:
        """Swap in the volume for echo ``index``, keeping plane, slice and window.

        The displayed plane and slice index are deliberately preserved so that
        moving along TE shows the same anatomy changing contrast, rather than
        jumping somewhere else in the volume.

        Every failure mode is caught here. An exception escaping this slot
        would make PyQt5 abort the whole process, so a bad echo has to end in a
        dialog, never in the window disappearing.
        """
        series = getattr(self, "_series", None)
        if series is None:
            return

        try:
            volume = series.volume(index)
        except Exception as exc:  # noqa: BLE001 - see docstring
            traceback.print_exc()
            QMessageBox.critical(
                self,
                "Could not load echo time",
                f"{series.echotime(index).label} could not be loaded.\n\n"
                f"{type(exc).__name__}: {exc}",
            )
            self._update_echo_label()
            return

        self._echo_index = index
        geometry_changed = self._volume is None or volume.shape != self._volume.shape
        self._volume = volume

        try:
            if geometry_changed:
                # A different slice count for this echo: reset the slice slider
                # rather than leaving it pointing past the end of the volume.
                self._image = None
                self._axes.clear()
                self._axes.set_axis_off()
                self._update_slice_range(reset_to_middle=True)
            if self._auto_window_check.isChecked():
                window = default_window(volume.data)
                self._window = window
                self._level_control.set_value(window.center)
                self._width_control.set_value(window.width)

            self._update_banner()
            self._update_info()
            self._update_echo_label()
            self._refresh()
        except Exception as exc:  # noqa: BLE001 - see docstring
            traceback.print_exc()
            QMessageBox.critical(
                self,
                "Could not display echo time",
                f"{series.echotime(index).label} loaded but could not be drawn.\n\n"
                f"{type(exc).__name__}: {exc}",
            )
            return
        series.prefetch_around(index)

    def _auto_window_toggled(self, _state: int) -> None:
        """Re-window immediately when per-echo auto-windowing is switched on."""
        if self._auto_window_check.isChecked():
            self._auto_window()

    def _preview_echo_label(self, index: int) -> None:
        """Show where the slider is *now*, before the volume has been loaded.

        Without this, dragging over uncached echoes would leave the caption
        showing the old echo time until the load finishes.
        """
        series = getattr(self, "_series", None)
        if series is None:
            return
        echo = series.echotime(index)
        state = "cached" if series.is_cached(index) else "loading..."
        self._echo_label.setText(
            f"{echo.label}  -  echo {index + 1} of {len(series)}  ({state})"
        )

    def _update_echo_label(self) -> None:
        """Refresh the caption under the echo slider."""
        series = getattr(self, "_series", None)
        if series is None:
            self._echo_label.setText("-")
            return
        echo = series.echotime(self._echo_index)
        slices = self._volume.shape[0] if self._volume is not None else "?"
        self._echo_label.setText(
            f"{echo.label}  -  echo {self._echo_index + 1} of {len(series)}"
            f"  -  {slices} slices"
        )

    # ------------------------------------------------------------------
    # Parent overrides that need echo context
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        """Draw as the parent does, then add the caption and the mask overlay."""
        super()._refresh()
        series = getattr(self, "_series", None)
        if series is None or self._volume is None:
            return

        self._draw_overlay()

        echo = series.echotime(self._echo_index)
        caption = (
            f"{series.name}  |  {self._volume.slice_caption(self._view, self._index)}"
            f"  |  {echo.label}"
        )
        self._axes.set_title(caption, color="#dddddd", fontsize=11)
        self._slice_label.setText(caption)
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Overlay rendering
    # ------------------------------------------------------------------

    def _label_plane(self) -> np.ndarray | None:
        """Cross-section of the cached label map matching the displayed plane.

        Returns None when there is no segmentation, or when the masks were
        computed on a volume of a different size - which would put them on the
        wrong voxels rather than simply looking wrong.
        """
        segmentation = getattr(self, "_segmentation", None)
        label_volume = getattr(self, "_label_volume", None)
        if segmentation is None or label_volume is None or self._volume is None:
            return None
        if not segmentation.matches(self._volume):
            return None
        # Same extraction path as the greyscale image, so the overlay cannot
        # drift out of alignment in coronal or sagittal views.
        return label_volume.extract(self._view, self._index).image

    def _draw_overlay(self) -> None:
        """Paint the selected masks as a half-transparent purple layer.

        The greyscale image underneath is never replaced: this is a second
        RGBA image on the same axes, transparent everywhere except on the
        selected labels.
        """
        self._clear_overlay_labels()

        plane = self._label_plane()
        selected = self._selected_segment_indices() if plane is not None else set()

        if plane is None or not selected:
            if self._overlay_image is not None and self._overlay_image in self._axes.images:
                self._overlay_image.set_visible(False)
            return

        mask = np.isin(plane, list(selected))
        rgba = np.zeros((*plane.shape, 4), dtype=np.float32)
        rgba[mask, 0] = OVERLAY_RGB[0]
        rgba[mask, 1] = OVERLAY_RGB[1]
        rgba[mask, 2] = OVERLAY_RGB[2]
        rgba[mask, 3] = OVERLAY_ALPHA

        # The parent clears the axes whenever the plane geometry changes, which
        # destroys the overlay artist; recreate it whenever it is not attached.
        stale = (
            self._overlay_image is None
            or self._overlay_image not in self._axes.images
            or self._overlay_image.get_array().shape[:2] != plane.shape
        )
        if stale:
            self._overlay_image = self._axes.imshow(
                rgba, interpolation="nearest", origin="upper", zorder=5
            )
            if self._image is not None:
                self._overlay_image.set_extent(self._image.get_extent())
        else:
            self._overlay_image.set_data(rgba)
        self._overlay_image.set_visible(True)

        self._draw_overlay_labels(plane, selected)

    def _clear_overlay_labels(self) -> None:
        """Remove the index numbers drawn on the overlay."""
        for text in getattr(self, "_overlay_texts", []):
            try:
                text.remove()
            except (NotImplementedError, ValueError):
                pass
        self._overlay_texts = []

    def _draw_overlay_labels(self, plane: np.ndarray, selected: set[int]) -> None:
        """Write each mask's number inside its own cross-section.

        The anchor is the deepest interior point of the region, found with a
        distance transform. A centroid would fall outside a ring or a crescent
        and land the number on a neighbouring mask - the same reasoning as the
        notebook's ``label_position``.
        """
        from scipy import ndimage as ndi

        for index in sorted(selected):
            region = plane == index
            if not region.any():
                continue  # This mask does not reach the current slice.
            padded = np.pad(region, 1)
            distance = ndi.distance_transform_edt(padded)
            y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
            self._overlay_texts.append(
                self._axes.text(
                    float(x - 1),
                    float(y - 1),
                    str(index),
                    color="#ffffff",
                    fontsize=9,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    zorder=6,
                    # Dark outline so the number stays readable over both the
                    # purple overlay and a bright patch of tissue.
                    path_effects=[withStroke(linewidth=2, foreground="#3b0a52")],
                )
            )

    def _update_info(self) -> None:
        """Add sample and echo-range detail to the parent's summary label."""
        super()._update_info()
        series = getattr(self, "_series", None)
        if series is None or self._volume is None:
            return
        k, j, i = self._volume.shape
        sz, sy, sx = self._volume.spacing
        self._info_label.setText(
            f"<b>{series.name}</b> - {len(series)} echo times "
            f"({series.echotime(0).value:g}-{series.echotime(-1).value:g} ms)<br>"
            f"{k} x {j} x {i} voxels<br>"
            f"{sz:.3g} x {sy:.3g} x {sx:.3g} mm<br>"
            f"{self._volume.description}"
        )

    def _default_export_name(self) -> str:
        """Include the sample and echo time in the suggested filename."""
        series = getattr(self, "_series", None)
        if series is None:
            return super()._default_export_name()
        echo = series.echotime(self._echo_index)
        return (
            f"{series.name}_TE{echo.value:g}ms_"
            f"{self._view.value.lower()}_{self._index + 1:03d}.svg"
        )

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        """Stop the prefetch thread before the window goes away."""
        series = getattr(self, "_series", None)
        if series is not None:
            series.close()
        super().closeEvent(event)


#: Strong references to open windows.
#:
#: PyQt keeps no Python reference to a parentless top-level widget, so a window
#: whose last Python reference goes away is destroyed - taking the application
#: with it once it was the only window. The window is not freed by refcounting,
#: because it and its child widgets reference each other through their signal
#: connections; it survives until the *cyclic* garbage collector happens to run.
#: This viewer allocates a 22 MB volume plus 128 datasets per echo load, which
#: drives collection hard, so the window would vanish mid-session at
#: unpredictable moments - a clean exit with no traceback and no crash report.
#: Holding the reference here is what prevents that.
_OPEN_WINDOWS: list["EchoViewerWindow"] = []


def create_viewer_with_echotime_slider(
    volumes_root: str | Path,
    sample: str | None = None,
    cache_size: int = DEFAULT_CACHE_SIZE,
    show: bool = True,
) -> EchoViewerWindow:
    """Build the echo-time viewer window.

    Assumes a ``QApplication`` already exists, or creates one. The window is
    registered in :data:`_OPEN_WINDOWS`, so it stays alive even if the caller
    discards the return value.

    Args:
        volumes_root: The ``Volumes`` directory.
        sample: Sample to open first; defaults to the first found.
        cache_size: Volumes held in memory per sample.
        show: Call ``show()`` on the window before returning it.

    Returns:
        The window, so callers can drive it (tests) or just display it.
    """
    QApplication.instance() or QApplication([])
    install_exception_logging()
    window = EchoViewerWindow(volumes_root, sample=sample, cache_size=cache_size)
    window.setWindowTitle("DICOM Echo-Time Viewer")
    window.resize(1240, 820)
    _OPEN_WINDOWS.append(window)
    window.destroyed.connect(lambda *_: _forget_window(window))
    if show:
        window.show()
    return window


def _forget_window(window: "EchoViewerWindow") -> None:
    """Drop our reference once Qt has destroyed the window."""
    try:
        _OPEN_WINDOWS.remove(window)
    except ValueError:
        pass


def run(
    volumes_root: str | Path,
    sample: str | None = None,
    cache_size: int = DEFAULT_CACHE_SIZE,
) -> int:
    """Launch the viewer and block until the window closes.

    Returns:
        The Qt exit code.
    """
    app = QApplication.instance() or QApplication([])
    # Bound to a local as well as registered, so the window cannot be collected
    # while the event loop is running.
    window = create_viewer_with_echotime_slider(
        volumes_root, sample=sample, cache_size=cache_size
    )
    exit_code = app.exec_()
    window.close()
    return exit_code
