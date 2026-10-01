"""Discovering samples and echo times inside a ``Volumes`` directory.

Expected layout::

    Volumes/
      Syringes/          <- sample
        8.0/             <- echo time, one folder per TE
          slice_0001.dcm <- all slices of that volume
          ...
        16.0/
      Pill1/
      ...

Echo-time folders are identified by name. Both conventions work: a bare
number (``16.0``) and a prefixed name (``EchoTime_2``). Where the DICOM files
themselves carry an ``EchoTime`` tag, that value wins over the folder name,
because the tag is what the scanner actually recorded.

Volume stacking is delegated to the original :mod:`dicomview` loader, so slice
sorting, spacing and rescaling behave identically to the existing viewer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from . import _bootstrap  # noqa: F401  (puts dicomview on sys.path)
from dicomview.loader import DicomLoadError, find_dicom_files
from dicomview.volume import Volume

#: Matches the first number in a folder name: "16.0" -> 16.0, "EchoTime_2" -> 2.
_NUMBER = re.compile(r"[-+]?\d*\.?\d+")

#: Folders that are never samples or echo times.
_IGNORED = {".DS_Store", "__MACOSX"}


@dataclass(frozen=True, order=True)
class EchoTime:
    """One echo-time folder within a sample.

    Attributes:
        value: Echo time in milliseconds. Taken from the DICOM ``EchoTime``
            tag when available, otherwise parsed from the folder name.
        path: The folder holding this volume's slices.
        from_tag: True when :attr:`value` came from the DICOM tag rather than
            from the folder name.
    """

    value: float
    path: Path
    from_tag: bool = False

    @property
    def label(self) -> str:
        """Human-readable label, e.g. ``"TE = 16 ms"``."""
        return f"TE = {self.value:g} ms"

    @property
    def name(self) -> str:
        """The folder name, useful for filenames and messages."""
        return self.path.name


def _numeric_hint(name: str) -> float | None:
    """Parse an echo time out of a folder name, or None if there is no number."""
    match = _NUMBER.search(name)
    return float(match.group()) if match else None


def _echo_time_from_dicom(folder: Path) -> float | None:
    """Read ``EchoTime`` from the first DICOM in ``folder``, if any.

    Only the header is read, so this stays cheap enough to call for every
    echo-time folder while building the list.
    """
    import pydicom

    files = find_dicom_files(folder, recursive=False)
    if not files:
        return None
    try:
        ds = pydicom.dcmread(str(files[0]), stop_before_pixels=True)
        echo = getattr(ds, "EchoTime", None)
        return float(echo) if echo is not None else None
    except Exception:
        return None


def _contains_dicom(folder: Path) -> bool:
    """True if ``folder`` directly holds at least one DICOM file."""
    return bool(find_dicom_files(folder, recursive=False))


def get_available_samples(volumes_root: str | Path) -> list[str]:
    """List sample names under a ``Volumes`` directory.

    A sample is any subdirectory that itself contains at least one echo-time
    folder, which keeps stray files and folders out of the list.

    Args:
        volumes_root: The ``Volumes`` directory.

    Returns:
        Sample names sorted alphabetically, e.g. ``["3D", "Pill1", ...]``.

    Raises:
        DicomLoadError: If ``volumes_root`` does not exist or holds no samples.
    """
    root = Path(volumes_root)
    if not root.is_dir():
        raise DicomLoadError(f"Not a directory: {root}")

    samples = [
        entry.name
        for entry in sorted(root.iterdir())
        if entry.is_dir() and entry.name not in _IGNORED and get_available_echotimes(entry)
    ]
    if not samples:
        raise DicomLoadError(
            f"No samples found in {root}. Expected subfolders such as "
            "Volumes/Syringes/16.0/*.dcm"
        )
    return samples


def get_available_echotimes(sample_path: str | Path) -> list[EchoTime]:
    """List the echo times available for one sample, sorted by echo time.

    Sorting is numeric, not lexical - lexical order would put ``104.0``
    before ``16.0`` and scramble the decay curve.

    Args:
        sample_path: A sample folder, e.g. ``Volumes/Syringes``.

    Returns:
        :class:`EchoTime` entries in ascending order. Empty if the folder
        holds no echo-time subfolders with DICOM data.
    """
    sample = Path(sample_path)
    if not sample.is_dir():
        return []

    entries: list[EchoTime] = []
    for folder in sorted(sample.iterdir()):
        if not folder.is_dir() or folder.name in _IGNORED:
            continue
        if not _contains_dicom(folder):
            continue

        tag_value = _echo_time_from_dicom(folder)
        if tag_value is not None:
            entries.append(EchoTime(value=tag_value, path=folder, from_tag=True))
            continue

        hint = _numeric_hint(folder.name)
        if hint is not None:
            entries.append(EchoTime(value=hint, path=folder, from_tag=False))

    # Fall back to name order only if nothing yielded a usable number.
    entries.sort(key=lambda e: (e.value, e.path.name))
    return entries


def load_volume_from_folder(echotime_path: str | Path) -> Volume:
    """Stack every DICOM slice in one echo-time folder into a 3D volume.

    Delegates to the original :mod:`dicomview` loader, so slices are sorted
    along the slice normal (not by filename) and spacing comes from the actual
    slice positions.

    Args:
        echotime_path: A folder holding the slices of a single volume.

    Returns:
        The stacked :class:`~dicomview.volume.Volume`.

    Raises:
        DicomLoadError: If the folder holds no readable DICOM slices, or holds
            more than one distinct series.
    """
    from dicomview.loader import load_directory

    folder = Path(echotime_path)
    volumes = load_directory(folder, recursive=False)
    if len(volumes) > 1:
        # One echo-time folder is meant to be exactly one volume. Say so
        # rather than silently picking the first of several.
        raise DicomLoadError(
            f"{folder} holds {len(volumes)} distinct series "
            f"({', '.join(v.description for v in volumes)}); expected one."
        )
    return volumes[0]


@lru_cache(maxsize=64)
def describe_sample(sample_path: str) -> str:
    """A short description of a sample, read from its first DICOM header.

    Cached because it is only ever used for labels.
    """
    echoes = get_available_echotimes(sample_path)
    if not echoes:
        return ""
    import pydicom

    files = find_dicom_files(echoes[0].path, recursive=False)
    if not files:
        return ""
    try:
        ds = pydicom.dcmread(str(files[0]), stop_before_pixels=True)
        return str(getattr(ds, "SeriesDescription", "") or "")
    except Exception:
        return ""
