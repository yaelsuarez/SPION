"""Writing computed segmentations to disk, sparse and dense.

This module only *saves*. It never segments: every function here takes an
already-computed :class:`~echoviewer.segmentation.Segmentation` and reuses its
cached masks, which is what keeps the masks identical to the ones on screen and
identical across every echo time.

Layout produced under ``output_root``::

    <Sample>/
      metadata.json                       sample-level: shape, spacing, TEs
      segmentations/                      SPARSE - mask voxels only
        Segmentation_1/
          voxels.csv.gz                   x, y, z, EchoTime, intensity
          metadata.json
      volumes/                            DENSE - reconstructable volumes
        Segmentation_1/
          mask.npz                        the boolean mask, once
          8.0/volume.npz                  one folder per echo time,
          16.0/volume.npz                 mirroring the source layout
          ...

**Sparse** holds one row per (voxel, echo time) and contains *only* voxels
inside the mask - nothing outside it, and no zero-filled background. It is the
form to load for fitting a relaxation curve, because it is already a table of
intensity against echo time.

**Dense** holds full-size volumes with the original intensity inside the mask
and 0 outside, one per echo time. It is the form to load for display or for any
operation that needs the surrounding geometry.

Either form reconstructs the other: the sparse table plus the shape in the
metadata scatters back into a dense volume, and the dense volume plus the mask
gathers back into the sparse table. :func:`reconstruct_dense_from_sparse` does
the first, and the tests check the two agree voxel for voxel.

Axis convention, recorded in every metadata file so nothing has to be guessed:
volume arrays are indexed ``[k, j, i]``, and the exported columns are
``x = i`` (column), ``y = j`` (row), ``z = k`` (slice).
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .segmentation import Segmentation
from .series import EchoSeries

from dicomview.volume import Volume

#: Subfolder holding the sparse, mask-only tables.
SPARSE_DIRNAME = "segmentations"

#: Subfolder holding the dense, full-size volumes.
DENSE_DIRNAME = "volumes"

#: Columns of the sparse table, in order.
SPARSE_COLUMNS = ("x", "y", "z", "EchoTime", "intensity")

#: How array indices map onto the exported coordinate names.
AXIS_MAP = {"x": "column (i)", "y": "row (j)", "z": "slice (k)"}


def segment_dirname(index: int) -> str:
    """Folder name for one segmentation, e.g. ``"Segmentation_3"``."""
    return f"Segmentation_{index}"


@dataclass
class ExportReport:
    """What an export actually wrote.

    Attributes:
        sample: Sample name.
        root: The ``<output_root>/<Sample>`` folder.
        segments: Segment indices exported.
        echo_times: Echo times exported, in milliseconds.
        files: Every file written.
        bytes_written: Total size on disk.
        skipped: ``(echo_time, reason)`` for echo times that were not written.
    """

    sample: str
    root: Path
    segments: list[int] = field(default_factory=list)
    echo_times: list[float] = field(default_factory=list)
    files: list[Path] = field(default_factory=list)
    bytes_written: int = 0
    skipped: list[tuple[float, str]] = field(default_factory=list)

    def record(self, path: Path) -> None:
        """Note a written file and add its size to the total."""
        self.files.append(path)
        try:
            self.bytes_written += path.stat().st_size
        except OSError:
            pass

    @property
    def summary(self) -> str:
        """One-line human summary."""
        megabytes = self.bytes_written / 1e6
        text = (
            f"{self.sample}: {len(self.segments)} segmentations x "
            f"{len(self.echo_times)} echo times, {len(self.files)} files, "
            f"{megabytes:.1f} MB"
        )
        if self.skipped:
            text += f" ({len(self.skipped)} echo times skipped)"
        return text


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def _volume_metadata(volume: Volume, segmentation: Segmentation, series: EchoSeries) -> dict:
    """Everything needed to place the exported data back in space."""
    return {
        "sample": series.name,
        "shape": {
            "array_index_order": ["z", "y", "x"],
            "dimensions": [int(n) for n in segmentation.source_shape],
        },
        "spacing_mm": {
            "z": float(volume.spacing[0]),
            "y": float(volume.spacing[1]),
            "x": float(volume.spacing[2]),
        },
        "axis_map": AXIS_MAP,
        "echo_times_ms": [float(e.value) for e in series.echotimes],
        "echo_time_folders": [e.path.name for e in series.echotimes],
        "segmentation": {
            "computed_on_echo_time_ms": float(series.echotime(0).value),
            "preset": segmentation.preset,
            "threshold_method": segmentation.params.method,
            "n_segments": len(segmentation),
            "warnings": list(segmentation.warnings),
        },
        "source": {
            "series_description": volume.description,
            "path": str(series.echotimes[0].path.parent),
        },
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _segment_metadata(segment, volume: Volume, segmentation: Segmentation, series: EchoSeries) -> dict:
    """Per-segmentation metadata, a superset of the sample metadata."""
    metadata = _volume_metadata(volume, segmentation, series)
    k0, k1, j0, j1, i0, i1 = segment.bbox
    metadata["segment"] = {
        "index": int(segment.index),
        "name": segment.name,
        "voxels": int(segment.voxels),
        "centroid_zyx": [float(c) for c in segment.centroid],
        "bounding_box": {
            "z": [int(k0), int(k1)],
            "y": [int(j0), int(j1)],
            "x": [int(i0), int(i1)],
        },
    }
    return metadata


def _write_json(path: Path, payload: dict, report: ExportReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report.record(path)


# ---------------------------------------------------------------------------
# Sparse writing
# ---------------------------------------------------------------------------


class _SparseWriter:
    """Appends rows to one segmentation's sparse table, one echo time at a time.

    Written incrementally on purpose. A single segmentation of the ``Pill1``
    sample holds 450k voxels, which over 26 echo times is 11.7 million rows;
    building that as one table before writing costs hundreds of megabytes of
    memory for no benefit.

    Uses Parquet when ``pyarrow`` is installed and gzipped CSV otherwise. Both
    hold exactly the same columns, so downstream code does not care which it
    gets - :func:`load_sparse` reads either.
    """

    def __init__(self, path_stem: Path):
        self._parquet_writer = None
        self._csv_handle = None
        try:
            import pyarrow  # noqa: F401
            import pyarrow.parquet  # noqa: F401

            self.path = path_stem.with_suffix(".parquet")
            self._use_parquet = True
        except ImportError:
            self.path = Path(str(path_stem) + ".csv.gz")
            self._use_parquet = False

    def __enter__(self) -> "_SparseWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self._use_parquet:
            self._csv_handle = gzip.open(self.path, "wt", newline="", compresslevel=6)
            self._csv_handle.write(",".join(SPARSE_COLUMNS) + "\n")
        return self

    def append(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        echo_time: float,
        intensity: np.ndarray,
    ) -> None:
        """Add every voxel of one mask at one echo time."""
        if self._use_parquet:
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.table(
                {
                    "x": pa.array(x, pa.int32()),
                    "y": pa.array(y, pa.int32()),
                    "z": pa.array(z, pa.int32()),
                    "EchoTime": pa.array(
                        np.full(x.size, echo_time, dtype=np.float32), pa.float32()
                    ),
                    "intensity": pa.array(intensity.astype(np.float32), pa.float32()),
                }
            )
            if self._parquet_writer is None:
                self._parquet_writer = pq.ParquetWriter(
                    self.path, table.schema, compression="snappy"
                )
            self._parquet_writer.write_table(table)
            return

        # Gzipped CSV, written through pandas: its writer is C code and is
        # roughly an order of magnitude quicker than numpy.savetxt, which
        # formats row by row and dominates the export time otherwise.
        import pandas as pd

        pd.DataFrame(
            {
                "x": x.astype(np.int32),
                "y": y.astype(np.int32),
                "z": z.astype(np.int32),
                "EchoTime": np.full(x.size, echo_time, dtype=np.float32),
                "intensity": intensity.astype(np.float32),
            }
        ).to_csv(self._csv_handle, header=False, index=False, lineterminator="\n")

    def close(self) -> None:
        if self._parquet_writer is not None:
            self._parquet_writer.close()
            self._parquet_writer = None
        if self._csv_handle is not None:
            self._csv_handle.close()
            self._csv_handle = None

    def __exit__(self, *_exc) -> None:
        self.close()


def load_sparse(path: str | Path):
    """Read a sparse table back, from either Parquet or gzipped CSV.

    Returns:
        A pandas DataFrame with the columns in :data:`SPARSE_COLUMNS`.
    """
    import pandas as pd

    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Dense writing
# ---------------------------------------------------------------------------


def _write_dense(
    path: Path,
    coords: tuple[np.ndarray, np.ndarray, np.ndarray],
    values: np.ndarray,
    shape: tuple[int, int, int],
    report: ExportReport,
) -> None:
    """Write one full-size volume: intensity inside the mask, 0 outside.

    Saved compressed. The array is mostly zeros, so compression takes it from
    21 MB down to roughly the size of the mask itself.
    """
    dense = np.zeros(shape, dtype=np.float32)
    dense[coords] = values
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, volume=dense)
    report.record(path)


def load_dense(path: str | Path) -> np.ndarray:
    """Read a dense volume written by :func:`export_sample`."""
    with np.load(Path(path)) as archive:
        return archive["volume"]


def load_mask(path: str | Path) -> np.ndarray:
    """Read a boolean mask written by :func:`export_sample`."""
    with np.load(Path(path)) as archive:
        return archive["mask"]


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


def reconstruct_dense_from_sparse(
    sparse_path: str | Path, metadata_path: str | Path, echo_time: float
) -> np.ndarray:
    """Rebuild a dense volume from the sparse table plus its metadata.

    This is the round trip that makes the sparse form self-sufficient: the
    table carries the coordinates and the metadata carries the shape, so the
    full spatial context can be restored without the dense files.

    Args:
        sparse_path: The segmentation's sparse table.
        metadata_path: The ``metadata.json`` beside it.
        echo_time: Which echo time to rebuild, in milliseconds.

    Returns:
        A float32 volume with intensity inside the mask and 0 elsewhere.
    """
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    shape = tuple(metadata["shape"]["dimensions"])

    frame = load_sparse(sparse_path)
    rows = frame[np.isclose(frame["EchoTime"], echo_time)]

    dense = np.zeros(shape, dtype=np.float32)
    # Columns are named x/y/z; the array is indexed [z, y, x].
    dense[
        rows["z"].to_numpy(dtype=np.intp),
        rows["y"].to_numpy(dtype=np.intp),
        rows["x"].to_numpy(dtype=np.intp),
    ] = rows["intensity"].to_numpy(dtype=np.float32)
    return dense


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def export_sample(
    series: EchoSeries,
    segmentation: Segmentation,
    output_root: str | Path,
    write_sparse: bool = True,
    write_dense: bool = True,
    write_images: bool = True,
    progress=None,
) -> ExportReport:
    """Write one sample's segmentations to disk, sparse and dense.

    The already-computed ``segmentation`` is reused as-is: this function never
    segments anything, so the exported masks are exactly the ones on screen,
    and the same voxels are used at every echo time.

    Each echo-time volume is loaded once and both representations are written
    from it, so the source data is read a single time no matter how many
    segmentations there are.

    Args:
        series: The sample's echo series, used to load each echo volume.
        segmentation: Masks computed from the first echo time.
        output_root: The ``Segmentations`` folder; ``<Sample>/`` is created
            inside it.
        write_sparse: Write the mask-only tables.
        write_dense: Write the full-size volumes.
        progress: Optional callable ``(done, total, message)`` for UI feedback.

    Returns:
        An :class:`ExportReport`.

    Raises:
        ValueError: If ``segmentation`` holds no segments.
    """
    if len(segmentation) == 0:
        raise ValueError(f"{series.name} has no segmentations to export")

    sample_root = Path(output_root) / series.name
    report = ExportReport(sample=series.name, root=sample_root)
    report.segments = [s.index for s in segmentation.segments]

    shape = tuple(segmentation.source_shape)

    # Mask coordinates are computed once per segmentation and reused for every
    # echo time. This is the concrete form of "the masks are echo-invariant":
    # the same index arrays address every volume.
    coordinates = {
        segment.index: np.nonzero(segmentation.labels == segment.index)
        for segment in segmentation.segments
    }

    first_volume = series.volume(0)
    _write_json(
        sample_root / "metadata.json",
        _volume_metadata(first_volume, segmentation, series),
        report,
    )

    writers: dict[int, _SparseWriter] = {}
    try:
        if write_sparse:
            for segment in segmentation.segments:
                writer = _SparseWriter(
                    sample_root / SPARSE_DIRNAME / segment_dirname(segment.index) / "voxels"
                )
                writer.__enter__()
                writers[segment.index] = writer

        if write_dense:
            for segment in segmentation.segments:
                mask_path = (
                    sample_root / DENSE_DIRNAME / segment_dirname(segment.index) / "mask.npz"
                )
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                mask = np.zeros(shape, dtype=bool)
                mask[coordinates[segment.index]] = True
                np.savez_compressed(mask_path, mask=mask)
                report.record(mask_path)

        # Previews are rendered from the first echo time alone, so when no data
        # is being written there is nothing to walk the echo times for.
        total = len(series) if (write_sparse or write_dense) else 0
        for echo_index in range(total):
            echo = series.echotime(echo_index)
            if progress is not None:
                progress(echo_index, total, f"{series.name}: {echo.label}")

            volume = series.volume(echo_index)
            if not segmentation.matches(volume):
                # Refuse rather than write masks onto voxels they do not
                # describe; an echo folder with a different slice count is a
                # data problem, not something to paper over.
                report.skipped.append(
                    (
                        float(echo.value),
                        f"volume shape {tuple(volume.shape)} != mask shape {shape}",
                    )
                )
                continue

            data = volume.data
            for segment in segmentation.segments:
                coords = coordinates[segment.index]
                values = data[coords]

                if write_sparse:
                    writers[segment.index].append(
                        x=coords[2], y=coords[1], z=coords[0],
                        echo_time=float(echo.value), intensity=values,
                    )
                if write_dense:
                    _write_dense(
                        sample_root
                        / DENSE_DIRNAME
                        / segment_dirname(segment.index)
                        / echo.path.name
                        / "volume.npz",
                        coords,
                        values,
                        shape,
                        report,
                    )
            report.echo_times.append(float(echo.value))
    finally:
        for writer in writers.values():
            writer.close()

    # Per-segmentation metadata is written last, so it can name the file that
    # was actually produced (Parquet or CSV).
    for segment in segmentation.segments:
        metadata = _segment_metadata(segment, first_volume, segmentation, series)
        if segment.index in writers:
            writer = writers[segment.index]
            metadata["sparse_file"] = writer.path.name
            report.record(writer.path)
        metadata["dense_files"] = {
            echo.path.name: f"{echo.path.name}/volume.npz" for echo in series.echotimes
        }
        _write_json(
            sample_root
            / SPARSE_DIRNAME
            / segment_dirname(segment.index)
            / "metadata.json",
            metadata,
            report,
        )

    # Human-readable counterpart to the data above: one preview per
    # segmentation, rendered from the same masks, so a folder called
    # "Segmentation_3" can be identified without loading anything.
    if write_images:
        from .preview import save_segment_previews

        if progress is not None:
            progress(len(series), len(series), f"{series.name}: rendering previews")
        for path in save_segment_previews(
            first_volume,
            segmentation,
            series.name,
            sample_root,
            series.echotime(0).label,
        ):
            report.record(path)

    if progress is not None:
        progress(len(series), len(series), report.summary)
    return report


def export_all(
    volumes_root: str | Path,
    output_root: str | Path,
    samples: list[str] | None = None,
    preset: str | None = None,
    cache_size: int = 4,
    write_sparse: bool = True,
    write_dense: bool = True,
    write_images: bool = True,
    progress=None,
) -> list[ExportReport]:
    """Segment and export every sample under ``volumes_root``.

    Used by the command line. Each sample is segmented once, from its first
    echo time, and that one result is what gets written - the same contract the
    viewer follows.

    Args:
        volumes_root: The ``Volumes`` directory.
        output_root: Where to write the ``Segmentations`` tree.
        samples: Restrict to these sample names; None means all of them.
        preset: Segmentation preset; None uses ``Auto``.
        cache_size: Volumes held in memory while exporting. Kept small because
            the export walks the echo times once, in order.
        progress: Optional ``(done, total, message)`` callable.

    Returns:
        One report per sample.
    """
    from .dataset import get_available_samples
    from .segmentation import AUTO_PRESET, segment_volume

    wanted = samples or get_available_samples(volumes_root)
    reports: list[ExportReport] = []

    for name in wanted:
        series = EchoSeries.from_sample(volumes_root, name, cache_size=cache_size)
        try:
            segmentation = segment_volume(
                series.volume(0), preset=preset or AUTO_PRESET
            )
            reports.append(
                export_sample(
                    series,
                    segmentation,
                    output_root,
                    write_sparse=write_sparse,
                    write_dense=write_dense,
                    write_images=write_images,
                    progress=progress,
                )
            )
        finally:
            series.close()
    return reports
