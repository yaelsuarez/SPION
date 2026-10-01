"""Driving the voxel-wise fit over both data sources.

Two inputs, deliberately handled differently:

* **Segmentations** — the dense ``.npz`` volumes already written by
  :mod:`echoviewer.export`. These are full-size volumes with intensity inside
  the segmentation and 0 outside, so they are used exactly as they are: not
  cropped, not re-masked. The voxels to fit are simply the non-zero ones at the
  first echo time.
* **Complete volumes** — reconstructed from the DICOM slices. Here "valid"
  has to be decided, since most of the volume is air: a voxel is fitted when
  its first-echo intensity clears the Otsu level of that volume, the same
  threshold the segmentation pipeline uses to separate sample from background.

Echo times come from the folder names in both cases and are read as numbers,
never assumed evenly spaced, and each volume is matched to its own echo time
before any fitting happens.

Segmentation folder names are carried through to the output untouched, so
``Segmentation_6_0.2`` stays ``Segmentation_6_0.2`` and the concentration
label survives into the results.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .dataset import get_available_echotimes, get_available_samples
from .export import DENSE_DIRNAME
from .fitting import FitReport, build_maps, fit_signals, summarise, write_fit_outputs
from .series import EchoSeries

#: Folder names under a sample's ``volumes/`` directory that hold a segmentation.
_SEGMENT_PATTERN = re.compile(r"^Segmentation_(\d+)(?:_(.*))?$")

#: Matches an echo-time folder name such as ``8.0`` or ``104.0``.
_NUMBER = re.compile(r"[-+]?\d*\.?\d+")


def parse_segmentation_name(name: str) -> tuple[int, str] | None:
    """Split ``Segmentation_6_0.2`` into ``(6, "0.2")``.

    The label is whatever follows the number - a concentration, ``water``,
    ``tissue`` - and is returned as written. Returns None if the folder is not
    a segmentation.
    """
    match = _SEGMENT_PATTERN.match(name)
    if match is None:
        return None
    return int(match.group(1)), (match.group(2) or "")


def find_segmentations(sample_root: str | Path) -> list[Path]:
    """Segmentation folders under ``<sample>/volumes/``, in numeric order."""
    volumes_dir = Path(sample_root) / DENSE_DIRNAME
    if not volumes_dir.is_dir():
        return []
    found = []
    for folder in volumes_dir.iterdir():
        parsed = parse_segmentation_name(folder.name)
        if folder.is_dir() and parsed is not None:
            found.append((parsed[0], folder))
    return [folder for _, folder in sorted(found, key=lambda pair: pair[0])]


def find_echo_folders(segmentation_dir: str | Path) -> list[tuple[float, Path]]:
    """Echo-time folders inside one segmentation, sorted by echo time.

    Sorted numerically, not lexically: ``104.0`` must not come before ``16.0``,
    or every decay curve is scrambled.
    """
    folders: list[tuple[float, Path]] = []
    for entry in Path(segmentation_dir).iterdir():
        if not entry.is_dir():
            continue
        number = _NUMBER.search(entry.name)
        if number is None:
            continue
        if any(entry.glob("*.npz")):
            folders.append((float(number.group()), entry))
    return sorted(folders, key=lambda pair: pair[0])


def _load_npz_volume(folder: Path) -> np.ndarray:
    """Read the dense volume from an echo-time folder."""
    files = sorted(folder.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no .npz in {folder}")
    with np.load(files[0]) as archive:
        key = "volume" if "volume" in archive else archive.files[0]
        return np.asarray(archive[key])


def _read_sample_metadata(sample_root: Path) -> dict:
    """The export's sample metadata, for spacing and provenance."""
    import json

    path = Path(sample_root) / "metadata.json"
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return {}


# ---------------------------------------------------------------------------
# Segmentations
# ---------------------------------------------------------------------------


def fit_segmentation(
    segmentation_dir: str | Path,
    output_dir: str | Path,
    sample: str,
    workers: int | None = None,
    progress=None,
) -> FitReport:
    """Fit every voxel of one saved segmentation.

    The ``.npz`` volumes are used directly. Their zeros are the region outside
    the segmentation and are never fitted; the spatial shape is preserved
    end to end.

    Args:
        segmentation_dir: ``<Segmentations>/<sample>/volumes/<Segmentation_N_label>``.
        output_dir: Where to write this segmentation's results.
        sample: Sample name, for the metadata.
        workers: Processes for the fit.
        progress: Optional ``(done, total, message)`` callable.

    Returns:
        A :class:`~echoviewer.fitting.FitReport`.

    Raises:
        FileNotFoundError: If the folder holds no echo-time volumes.
    """
    segmentation_dir = Path(segmentation_dir)
    echoes = find_echo_folders(segmentation_dir)
    if not echoes:
        raise FileNotFoundError(f"no echo-time volumes in {segmentation_dir}")

    echo_times = [value for value, _ in echoes]
    first = _load_npz_volume(echoes[0][1])
    shape = first.shape

    # The non-zero voxels of the first echo *are* the segmentation. No second
    # mask is built, and the volume is not cropped.
    coordinates = np.nonzero(first != 0)
    n_voxels = coordinates[0].size

    signals = np.zeros((n_voxels, len(echoes)), dtype=np.float32)
    signals[:, 0] = first[coordinates]
    for column, (_, folder) in enumerate(echoes[1:], start=1):
        volume = _load_npz_volume(folder)
        if volume.shape != shape:
            raise ValueError(
                f"{folder} has shape {volume.shape}, expected {shape}"
            )
        # Same coordinates at every echo time: the mask is echo-invariant.
        signals[:, column] = volume[coordinates]

    results = fit_signals(echo_times, signals, workers=workers, progress=progress)
    maps = build_maps(results, coordinates, shape)

    parsed = parse_segmentation_name(segmentation_dir.name)
    sample_metadata = _read_sample_metadata(segmentation_dir.parent.parent)
    spacing = sample_metadata.get("spacing_mm", {})
    metadata = {
        "shape": [int(n) for n in shape],
        "echo_times": echo_times,
        "voxel_spacing": [spacing.get("z"), spacing.get("y"), spacing.get("x")],
        "source": "segmentation",
        "segmentation_name": segmentation_dir.name,
        "segmentation_number": parsed[0] if parsed else None,
        "segmentation_label": parsed[1] if parsed else "",
        "sample_name": sample,
        "shape_order": ["z", "y", "x"],
        "axis_map": {"x": "column (i)", "y": "row (j)", "z": "slice (k)"},
        "n_voxels_evaluated": int(n_voxels),
        "valid_voxel_rule": "non-zero in the first echo .npz (the segmentation itself)",
        "models": {
            "noB": "I0 * exp(-rs * TE)",
            "B": "I0 * exp(-rs * TE) + B",
        },
        "model_choice": {"0": "not fitted", "1": "Model A (noB)", "2": "Model B"},
        "model_choice_criterion": "lower AIC; ties go to Model A",
        "source_path": str(segmentation_dir),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if sample_metadata.get("source"):
        metadata["dicom_source"] = sample_metadata["source"]

    write_fit_outputs(output_dir, maps, coordinates, results, metadata)
    return summarise(
        f"{sample}/{segmentation_dir.name}", "segmentation",
        Path(output_dir), shape, echo_times, results,
    )


# ---------------------------------------------------------------------------
# Complete volumes
# ---------------------------------------------------------------------------


def _foreground_mask(volume: np.ndarray) -> np.ndarray:
    """Voxels worth fitting in a complete volume.

    Most of a complete volume is air, and fitting noise wastes hours and
    produces meaningless parameters. The cut is the Otsu level of the first
    echo - the same criterion the segmentation pipeline uses to separate
    sample from background, so the two stay consistent.
    """
    from skimage.filters import threshold_otsu

    finite = volume[np.isfinite(volume)]
    if finite.size == 0 or float(finite.max()) <= float(finite.min()):
        return np.zeros(volume.shape, dtype=bool)
    return volume > threshold_otsu(finite.ravel())


def fit_sample_volume(
    volumes_root: str | Path,
    sample: str,
    output_dir: str | Path,
    workers: int | None = None,
    progress=None,
) -> FitReport:
    """Fit every valid voxel of one sample's complete volume.

    Args:
        volumes_root: The ``Volumes`` directory of DICOM data.
        sample: Sample name.
        output_dir: Where to write this sample's results.
        workers: Processes for the fit.
        progress: Optional ``(done, total, message)`` callable.

    Returns:
        A :class:`~echoviewer.fitting.FitReport`.

    Raises:
        ValueError: If the echo volumes do not share one shape.
    """
    series = EchoSeries.from_sample(volumes_root, sample, cache_size=2)
    try:
        echo_times = [echo.value for echo in series.echotimes]
        first_volume = series.volume(0)
        shape = first_volume.shape
        spacing = first_volume.spacing

        coordinates = np.nonzero(_foreground_mask(first_volume.data))
        n_voxels = coordinates[0].size

        signals = np.zeros((n_voxels, len(series)), dtype=np.float32)
        signals[:, 0] = first_volume.data[coordinates]
        for column in range(1, len(series)):
            volume = series.volume(column)
            if volume.shape != shape:
                raise ValueError(
                    f"{sample} echo {echo_times[column]} has shape "
                    f"{volume.shape}, expected {shape}"
                )
            signals[:, column] = volume.data[coordinates]

        results = fit_signals(echo_times, signals, workers=workers, progress=progress)
        maps = build_maps(results, coordinates, shape)

        metadata = {
            "shape": [int(n) for n in shape],
            "echo_times": [float(t) for t in echo_times],
            "voxel_spacing": [float(s) for s in spacing],
            "source": "volume",
            "sample_name": sample,
            "shape_order": ["z", "y", "x"],
            "axis_map": {"x": "column (i)", "y": "row (j)", "z": "slice (k)"},
            "series_description": first_volume.description,
            "n_voxels_evaluated": int(n_voxels),
            "valid_voxel_rule": "first-echo intensity above the volume's Otsu level",
            "models": {
                "noB": "I0 * exp(-rs * TE)",
                "B": "I0 * exp(-rs * TE) + B",
            },
            "model_choice": {"0": "not fitted", "1": "Model A (noB)", "2": "Model B"},
            "model_choice_criterion": "lower AIC; ties go to Model A",
            "source_path": str(Path(volumes_root) / sample),
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        write_fit_outputs(output_dir, maps, coordinates, results, metadata)
        return summarise(sample, "volume", Path(output_dir), shape, echo_times, results)
    finally:
        series.close()


# ---------------------------------------------------------------------------
# Whole-dataset driver
# ---------------------------------------------------------------------------


def run_segmentation_fits(
    segmentations_root: str | Path,
    output_root: str | Path,
    samples: list[str] | None = None,
    workers: int | None = None,
    progress=None,
) -> list[FitReport]:
    """Fit every saved segmentation of every sample.

    Output goes to ``<output_root>/<sample>/<Segmentation_N_label>/``, with the
    segmentation folder name preserved exactly.
    """
    root = Path(segmentations_root)
    names = samples or [
        folder.name
        for folder in sorted(root.iterdir())
        if folder.is_dir() and find_segmentations(folder)
    ]

    reports: list[FitReport] = []
    for sample in names:
        for segmentation_dir in find_segmentations(root / sample):
            if progress is not None:
                progress(0, 1, f"{sample}/{segmentation_dir.name}")
            reports.append(
                fit_segmentation(
                    segmentation_dir,
                    Path(output_root) / sample / segmentation_dir.name,
                    sample,
                    workers=workers,
                )
            )
    return reports


def run_volume_fits(
    volumes_root: str | Path,
    output_root: str | Path,
    samples: list[str] | None = None,
    workers: int | None = None,
    progress=None,
) -> list[FitReport]:
    """Fit every sample's complete volume.

    Output goes to ``<output_root>/<sample>/``.
    """
    names = samples or get_available_samples(volumes_root)
    reports: list[FitReport] = []
    for sample in names:
        if progress is not None:
            progress(0, 1, f"{sample} (complete volume)")
        reports.append(
            fit_sample_volume(
                volumes_root, sample, Path(output_root) / sample, workers=workers
            )
        )
    return reports


def echo_times_for(volumes_root: str | Path, sample: str) -> list[float]:
    """Echo times detected for one sample, for reporting."""
    return [echo.value for echo in get_available_echotimes(Path(volumes_root) / sample)]
