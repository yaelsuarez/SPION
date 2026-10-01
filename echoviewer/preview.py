"""Rendering preview images of a segmentation, for human identification.

The data written by :mod:`echoviewer.export` is precise but unreadable: a
folder called ``Segmentation_3`` says nothing about which compartment it holds.
This module renders one PNG per segmentation showing the mask in the context of
the volume, so a person can tell at a glance which is which.

Deliberately identical to the viewer's overlay, not merely similar: the colour
and opacity live here as the single definition that :mod:`echoviewer.ui`
imports, and the planes are cut with the viewer's own
:meth:`~dicomview.volume.Volume.extract`. There is no second implementation to
drift out of step.

Rendering goes through ``Figure`` with the Agg canvas rather than ``pyplot``,
so it works headless, does not touch whatever backend the GUI has selected, and
is safe to call while a viewer window is open.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patheffects import withStroke

from .segmentation import Segmentation

from dicomview.volume import View, Volume
from dicomview.windowing import WindowLevel, default_window, normalize

#: Overlay colour and opacity for segmentation masks. Defined here and imported
#: by the viewer, so the saved images and the screen cannot disagree.
OVERLAY_RGB = (0.62, 0.21, 0.85)
OVERLAY_ALPHA = 0.5

#: Subfolder holding the preview images, beside ``segmentations`` and ``volumes``.
IMAGES_DIRNAME = "images_segmentations"

_BACKGROUND = "#111111"
_FOREGROUND = "#dddddd"


def preview_filename(sample: str, segment_index: int) -> str:
    """File name for one segmentation's preview, e.g. ``Pill2_Segmentation_1.png``."""
    return f"{sample}_Segmentation_{segment_index}.png"


def _busiest_slice(mask: np.ndarray, view: View) -> int:
    """Index of the plane holding the most mask voxels.

    A representative slice has to be chosen somehow, and the fullest cut
    through the region shows the most of its shape. Picking the middle of the
    bounding box would land on a thin edge for anything curved.
    """
    axes = {View.AXIAL: (1, 2), View.CORONAL: (0, 2), View.SAGITTAL: (0, 1)}[view]
    counts = mask.sum(axis=axes)
    return int(np.argmax(counts)) if counts.any() else mask.shape[0] // 2


def _draw_panel(axes, plane_image, label_plane, segment_index: int, aspect: float) -> None:
    """Draw one orthogonal view: greyscale underneath, purple mask on top."""
    axes.imshow(
        plane_image, cmap="gray", vmin=0.0, vmax=1.0,
        interpolation="nearest", aspect=aspect, origin="upper",
    )

    mask = label_plane == segment_index
    rgba = np.zeros((*label_plane.shape, 4), dtype=np.float32)
    rgba[mask, 0], rgba[mask, 1], rgba[mask, 2] = OVERLAY_RGB
    rgba[mask, 3] = OVERLAY_ALPHA
    axes.imshow(rgba, interpolation="nearest", aspect=aspect, origin="upper", zorder=5)

    if mask.any():
        # Number the region at the deepest interior point of this cross-section,
        # as the viewer does; a centroid can fall outside a curved shape.
        from scipy import ndimage as ndi

        distance = ndi.distance_transform_edt(np.pad(mask, 1))
        y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
        axes.text(
            float(x - 1), float(y - 1), str(segment_index),
            color="#ffffff", fontsize=9, fontweight="bold",
            ha="center", va="center", zorder=6,
            path_effects=[withStroke(linewidth=2, foreground="#3b0a52")],
        )
    axes.set_axis_off()


def render_segment_preview(
    volume: Volume,
    segmentation: Segmentation,
    segment_index: int,
    sample: str,
    echo_label: str,
    window: WindowLevel | None = None,
) -> Figure:
    """Build the three-view preview figure for one segmentation.

    Args:
        volume: The volume to show underneath. Normally the first echo time,
            the one the masks were computed from.
        segmentation: Already-computed masks. Never recomputed here.
        segment_index: Which label to highlight.
        sample: Sample name, for the title.
        echo_label: Echo-time label, for the title, e.g. ``"TE = 8 ms"``.
        window: Display window. Pass one shared window for a whole sample so
            every segmentation's preview is on the same intensity scale;
            defaults to the volume's own percentile window.

    Returns:
        A matplotlib Figure with axial, coronal and sagittal panels.

    Raises:
        ValueError: If the masks do not fit ``volume``.
    """
    if not segmentation.matches(volume):
        raise ValueError(
            f"mask shape {tuple(segmentation.source_shape)} does not fit "
            f"volume shape {tuple(volume.shape)}"
        )

    window = window or default_window(volume.data)
    mask = segmentation.labels == segment_index
    label_volume = segmentation.as_volume()

    figure = Figure(figsize=(12, 5.2), facecolor=_BACKGROUND)
    FigureCanvasAgg(figure)
    panels = figure.subplots(1, 3)

    for axes, view in zip(panels, View):
        index = _busiest_slice(mask, view)
        # Both planes come from the viewer's own extraction, so the overlay
        # cannot be misaligned with the greyscale in any view.
        plane = volume.extract(view, index)
        labels = label_volume.extract(view, index)
        _draw_panel(
            axes, normalize(plane.image, window), labels.image, segment_index, plane.aspect
        )
        covered = int((labels.image == segment_index).sum())
        axes.set_title(
            f"{view.value} — slice {index} ({covered:,} px)",
            color=_FOREGROUND, fontsize=10,
        )

    voxels = int(mask.sum())
    figure.suptitle(
        f"{sample} — Segmentation {segment_index} — {echo_label} — {voxels:,} voxels",
        color=_FOREGROUND, fontsize=14, fontweight="600",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    return figure


def save_segment_previews(
    volume: Volume,
    segmentation: Segmentation,
    sample: str,
    sample_root: str | Path,
    echo_label: str,
    dpi: int = 110,
) -> list[Path]:
    """Render and save one preview per segmentation.

    Every preview of a sample shares one display window, derived from the
    volume as a whole, so brightness is comparable between them rather than
    each being stretched to its own range.

    Args:
        volume: Volume to show; normally the first echo time.
        segmentation: Already-computed masks.
        sample: Sample name.
        sample_root: The ``<output_root>/<Sample>`` folder. Images go into
            ``<sample_root>/images_segmentations/``.
        echo_label: Echo-time label for the titles.
        dpi: Output resolution.

    Returns:
        The paths written, in segmentation order.
    """
    folder = Path(sample_root) / IMAGES_DIRNAME
    folder.mkdir(parents=True, exist_ok=True)

    shared_window = default_window(volume.data)
    written: list[Path] = []
    for segment in segmentation.segments:
        figure = render_segment_preview(
            volume, segmentation, segment.index, sample, echo_label, window=shared_window
        )
        path = folder / preview_filename(sample, segment.index)
        figure.savefig(path, dpi=dpi, facecolor=_BACKGROUND)
        written.append(path)
    return written
