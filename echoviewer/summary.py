"""Mean intensity per segmentation per echo time, exported to one workbook.

Pure analysis: this reads the masks already written by :mod:`echoviewer.export`
and the source volumes, and writes a spreadsheet. It never segments, never
normalises, and never touches anything it reads.

The mean is taken over the mask voxels only::

    values = volume.data[mask]      # only voxels inside the segmentation
    mean = values.mean()

not over a zero-filled volume. That distinction matters: the dense exports are
0 outside the mask, and averaging those zeros in would drag every value towards
0 by the ratio of mask size to volume size - for a segmentation covering 0.4%
of the volume, a mean of 1315 would read as about 5.

An empty mask yields ``NaN`` rather than 0, because 0 is a plausible measured
intensity and would be indistinguishable from a real reading.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .export import DENSE_DIRNAME, load_mask, segment_dirname
from .series import EchoSeries

#: Default workbook name, written inside the segmentations root.
DEFAULT_WORKBOOK = "segmentation_mean_intensities.xlsx"

#: Preferred sheet order; anything else follows, in the order found on disk.
_SHEET_ORDER = ("Syringes", "3D", "Pill1", "Pill2")


def load_saved_masks(sample_root: str | Path) -> dict[int, np.ndarray]:
    """Load every ``mask.npz`` already written for one sample.

    Reusing the saved masks is what makes this step an analysis of the existing
    export rather than a second opinion about where the segmentations are.

    Args:
        sample_root: The ``<Segmentations>/<Sample>`` folder.

    Returns:
        ``{segmentation_index: boolean mask}``, empty if none are on disk.
    """
    volumes_dir = Path(sample_root) / DENSE_DIRNAME
    if not volumes_dir.is_dir():
        return {}

    masks: dict[int, np.ndarray] = {}
    for folder in sorted(volumes_dir.iterdir()):
        if not folder.is_dir() or not folder.name.startswith("Segmentation_"):
            continue
        mask_file = folder / "mask.npz"
        if not mask_file.is_file():
            continue
        try:
            index = int(folder.name.split("_")[-1])
        except ValueError:
            continue
        masks[index] = load_mask(mask_file)
    return masks


def mean_intensity_table(series: EchoSeries, masks: dict[int, np.ndarray]):
    """Build the echo-time x segmentation table of mean intensities.

    Args:
        series: The sample's echo series, used to load each echo volume.
        masks: ``{index: boolean mask}``, as saved by the export.

    Returns:
        A DataFrame whose first column is ``EchoTime`` and whose remaining
        columns are ``Segmentation_1 .. Segmentation_N`` in numeric order, one
        row per echo time, sorted ascending by echo time.

    Raises:
        ValueError: If ``masks`` is empty.
    """
    import pandas as pd

    if not masks:
        raise ValueError(f"no saved masks found for {series.name}")

    ordered = sorted(masks)
    rows = []
    for echo_index in range(len(series)):
        echo = series.echotime(echo_index)
        volume = series.volume(echo_index)
        # The echo time is carried through exactly as recorded; nothing here
        # averages, interpolates or rounds it.
        row: dict[str, float] = {"EchoTime": float(echo.value)}

        for index in ordered:
            mask = masks[index]
            column = segment_dirname(index)
            if mask.shape != volume.shape:
                # Cannot address these voxels in this volume; a blank is
                # honest, a number would not be.
                row[column] = np.nan
                continue
            values = volume.data[mask]
            row[column] = float(values.mean()) if values.size else np.nan
        rows.append(row)

    frame = pd.DataFrame(rows, columns=["EchoTime", *(segment_dirname(i) for i in ordered)])
    return frame.sort_values("EchoTime").reset_index(drop=True)


def _sheet_order(names: list[str]) -> list[str]:
    """Preferred samples first, then whatever else was found."""
    known = [name for name in _SHEET_ORDER if name in names]
    return known + [name for name in names if name not in known]


def build_tables(
    volumes_root: str | Path,
    segmentations_root: str | Path,
    samples: list[str] | None = None,
    cache_size: int = 3,
    progress=None,
) -> dict[str, "object"]:
    """Compute the table for every sample that has saved masks.

    Args:
        volumes_root: The ``Volumes`` directory holding the source DICOM.
        segmentations_root: The ``Segmentations`` directory holding the export.
        samples: Restrict to these samples; None means every sample with masks.
        cache_size: Volumes held in memory. Small on purpose - each echo time
            is visited once, in order.
        progress: Optional ``(done, total, message)`` callable.

    Returns:
        ``{sample: DataFrame}`` in sheet order.

    Raises:
        FileNotFoundError: If ``segmentations_root`` holds no saved masks.
    """
    root = Path(segmentations_root)
    if not root.is_dir():
        raise FileNotFoundError(f"no segmentations directory at {root}")

    available = [
        folder.name
        for folder in sorted(root.iterdir())
        if folder.is_dir() and load_saved_masks(folder)
    ]
    if samples:
        available = [name for name in available if name in samples]
    if not available:
        raise FileNotFoundError(
            f"no saved masks under {root}; run --export-segmentations first"
        )

    tables: dict[str, object] = {}
    names = _sheet_order(available)
    for position, name in enumerate(names, start=1):
        if progress is not None:
            progress(position, len(names), f"{name}: averaging inside masks")
        masks = load_saved_masks(root / name)
        series = EchoSeries.from_sample(volumes_root, name, cache_size=cache_size)
        try:
            tables[name] = mean_intensity_table(series, masks)
        finally:
            series.close()
    return tables


def write_workbook(tables: dict[str, object], path: str | Path) -> Path:
    """Write one sheet per sample to a single ``.xlsx``.

    Args:
        tables: ``{sample: DataFrame}``; insertion order becomes sheet order.
        path: Destination file. A ``.xlsx`` suffix is added if missing.

    Returns:
        The path written.
    """
    import pandas as pd

    destination = Path(path)
    if destination.suffix.lower() != ".xlsx":
        destination = destination.with_suffix(".xlsx")
    destination.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(destination, engine="openpyxl") as writer:
        for sample, frame in tables.items():
            # Excel sheet names are capped at 31 characters.
            frame.to_excel(writer, sheet_name=str(sample)[:31], index=False)
    return destination


def export_mean_intensities(
    volumes_root: str | Path,
    segmentations_root: str | Path,
    workbook: str | Path | None = None,
    samples: list[str] | None = None,
    progress=None,
) -> tuple[Path, dict[str, object]]:
    """Compute every sample's table and write the workbook.

    Args:
        volumes_root: The ``Volumes`` directory.
        segmentations_root: The ``Segmentations`` directory.
        workbook: Output file; defaults to :data:`DEFAULT_WORKBOOK` inside
            ``segmentations_root``.
        samples: Restrict to these samples.
        progress: Optional ``(done, total, message)`` callable.

    Returns:
        ``(path_written, {sample: DataFrame})``.
    """
    tables = build_tables(
        volumes_root, segmentations_root, samples=samples, progress=progress
    )
    destination = Path(workbook) if workbook else Path(segmentations_root) / DEFAULT_WORKBOOK
    return write_workbook(tables, destination), tables
