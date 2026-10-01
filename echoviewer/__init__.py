"""Echo-time viewer: the original DICOM viewer plus a TE axis.

Extends :mod:`dicomview` without modifying it. The original package is located
and imported at runtime by :mod:`echoviewer._bootstrap`.

Command line::

    python run_echo_viewer.py "/path/to/Volumes"
    python run_echo_viewer.py "/path/to/Volumes" --sample Syringes

From Python::

    from echoviewer import get_available_echotimes, load_volume_from_folder

    echoes = get_available_echotimes("Volumes/Syringes")
    volume = load_volume_from_folder(echoes[0].path)

The dataset helpers import no Qt, so they work in notebooks and scripts.
"""

from .dataset import (
    EchoTime,
    describe_sample,
    get_available_echotimes,
    get_available_samples,
    load_volume_from_folder,
)
from .export import (
    ExportReport,
    export_all,
    export_sample,
    load_dense,
    load_mask,
    load_sparse,
    reconstruct_dense_from_sparse,
)
from .segmentation import AUTO_PRESET, PRESETS, Segmentation, segment_volume
from .summary import (
    DEFAULT_WORKBOOK,
    build_tables,
    export_mean_intensities,
    load_saved_masks,
    mean_intensity_table,
    write_workbook,
)
from .series import DEFAULT_CACHE_SIZE, PREFETCH_RADIUS, EchoSeries

__version__ = "1.0.0"

__all__ = [
    "AUTO_PRESET",
    "DEFAULT_CACHE_SIZE",
    "DEFAULT_WORKBOOK",
    "build_tables",
    "export_mean_intensities",
    "load_saved_masks",
    "mean_intensity_table",
    "write_workbook",
    "PREFETCH_RADIUS",
    "PRESETS",
    "EchoSeries",
    "EchoTime",
    "ExportReport",
    "Segmentation",
    "describe_sample",
    "export_all",
    "export_sample",
    "get_available_echotimes",
    "get_available_samples",
    "load_dense",
    "load_mask",
    "load_sparse",
    "load_volume_from_folder",
    "reconstruct_dense_from_sparse",
    "segment_volume",
    # UI names are re-exported lazily by __getattr__ so that importing this
    # package in a headless script does not pull in Qt.
    "EchoViewerWindow",
    "create_viewer_with_echotime_slider",
    "run",
]

_UI_NAMES = {"EchoViewerWindow", "create_viewer_with_echotime_slider", "run"}


def __getattr__(name: str):
    """Import the Qt UI only when one of its names is actually used."""
    if name in _UI_NAMES:
        from . import ui

        return getattr(ui, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
