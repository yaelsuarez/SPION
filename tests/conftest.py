"""Fixtures: a synthetic Volumes tree, plus opt-in access to the real dataset."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pydicom  # noqa: E402
from pydicom.dataset import Dataset, FileMetaDataset  # noqa: E402
from pydicom.uid import ExplicitVRLittleEndian, generate_uid  # noqa: E402

MR_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.4"

#: Point this at the real Volumes folder to run the integration tests.
REAL_DATA_ENV = "ECHOVIEWER_TEST_VOLUMES"


def _save(ds: Dataset, path: Path) -> Path:
    try:
        ds.save_as(str(path), enforce_file_format=True)  # pydicom >= 3
    except TypeError:  # pragma: no cover - pydicom 2.x
        ds.save_as(str(path), write_like_original=False)
    return path


def _write_slice(
    folder: Path,
    index: int,
    echo_time: float,
    series_uid: str,
    value: int,
    pixels: np.ndarray | None = None,
) -> None:
    """Write one slice.

    By default the whole slice is filled with ``value``, so tests can read the
    volume and slice index straight back out of the pixel data. Pass ``pixels``
    to write real image content instead.
    """
    if pixels is not None:
        rows, cols = pixels.shape
    else:
        rows, cols = 6, 4
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = MR_IMAGE_STORAGE
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = MR_IMAGE_STORAGE
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID = series_uid
    ds.StudyInstanceUID = generate_uid()
    ds.Modality = "MR"
    ds.SeriesDescription = "Synthetic MSME"
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.Rows, ds.Columns = rows, cols
    ds.PixelSpacing = [0.5, 0.5]
    ds.SliceThickness = 0.5
    ds.EchoTime = echo_time
    ds.InstanceNumber = index + 1
    # Slices advance along X, as in the real dataset, so tests exercise
    # sorting by the slice normal rather than by Z. This orientation makes the
    # normal -X, so positions run negative to keep stack order == slice index.
    ds.ImageOrientationPatient = [0, 1, 0, 0, 0, -1]
    ds.ImagePositionPatient = [-0.5 * index, 0.0, 0.0]
    ds.RescaleSlope = 1.0
    ds.RescaleIntercept = 0.0
    content = pixels if pixels is not None else np.full((rows, cols), value, dtype=np.uint16)
    ds.PixelData = np.ascontiguousarray(content, dtype=np.uint16).tobytes()
    _save(ds, folder / f"slice_{index:03d}.dcm")


@pytest.fixture
def volumes_tree(tmp_path) -> Path:
    """A miniature Volumes/ tree: 2 samples x 3 echo times x 4 slices.

    Echo-time folders are named with bare numbers whose lexical order differs
    from their numeric order (8.0, 16.0, 104.0), so tests catch sorting bugs.
    Each slice's pixel value is ``100 * echo_index + slice_index``.
    """
    root = tmp_path / "Volumes"
    echo_times = [8.0, 16.0, 104.0]

    for sample in ("Syringes", "Pill1"):
        for echo_index, te in enumerate(echo_times):
            folder = root / sample / f"{te}"
            folder.mkdir(parents=True)
            series_uid = generate_uid()
            for slice_index in range(4):
                _write_slice(
                    folder,
                    slice_index,
                    te,
                    series_uid,
                    value=100 * echo_index + slice_index,
                )

    # Decoys that must be ignored by discovery.
    (root / "Configuration.xlsx").write_text("not dicom")
    (root / "Syringes" / "notes.txt").write_text("not dicom")
    return root


@pytest.fixture
def prefixed_tree(tmp_path) -> Path:
    """A Volumes tree using the ``EchoTime_N`` naming convention instead."""
    root = tmp_path / "Volumes"
    for echo_index, te in enumerate([12.0, 24.0]):
        folder = root / "Sample" / f"EchoTime_{echo_index + 1}"
        folder.mkdir(parents=True)
        series_uid = generate_uid()
        for slice_index in range(3):
            _write_slice(folder, slice_index, te, series_uid, value=slice_index)
    return root


@pytest.fixture
def ragged_tree(volumes_tree) -> Path:
    """A Volumes tree where one echo folder is an incomplete export.

    Mirrors the real ``Pill2/8.0``, which holds 21 slices where every other
    echo folder holds 128.
    """
    short = volumes_tree / "Syringes" / "16.0"
    for path in sorted(short.glob("*.dcm"))[2:]:
        path.unlink()
    return volumes_tree


def _bars_slice(rows: int, cols: int, scale: float, rng) -> np.ndarray:
    """One slice holding three separated bright bars, dimmed by ``scale``."""
    image = np.zeros((rows, cols), dtype=np.float32)
    image[5:15, 5:15] = 900.0
    image[25:35, 5:15] = 850.0
    image[15:25, 25:35] = 800.0
    image = image * scale + rng.normal(0.0, 3.0, image.shape)
    return np.clip(image, 0, None).astype(np.uint16)


@pytest.fixture
def segmentable_tree(tmp_path) -> Path:
    """A Volumes tree whose volumes contain three separable 3D bars.

    ``volumes_tree`` holds flat 6x4 images with nothing to find; the
    segmentation UI tests need a sample that actually produces masks. Signal
    decays with echo time here too, as in the real data, so the masks must come
    from the first echo to be usable at the last one.
    """
    root = tmp_path / "Volumes"
    rows, cols, n_slices = 40, 40, 8
    echo_times = [8.0, 16.0, 104.0]
    rng = np.random.default_rng(1)

    for sample in ("Syringes", "Pill1"):
        for echo_index, te in enumerate(echo_times):
            folder = root / sample / f"{te}"
            folder.mkdir(parents=True)
            series_uid = generate_uid()
            scale = 1.0 / (1 + echo_index)  # signal falls as TE rises
            for slice_index in range(n_slices):
                _write_slice(
                    folder, slice_index, te, series_uid, value=0,
                    pixels=_bars_slice(rows, cols, scale, rng),
                )
    return root


@pytest.fixture
def blobs_volume():
    """A synthetic volume holding three separated bars of known size.

    Each bar runs the full length of the stack axis, so a correct 3D
    segmentation gives one region per bar spanning every slice - which a
    slice-by-slice approach would instead label independently on each slice.
    """
    from dicomview.volume import Volume

    data = np.zeros((12, 40, 40), dtype=np.float32)
    data[:, 5:15, 5:15] = 900.0    # bar A, bright
    data[:, 25:35, 5:15] = 850.0   # bar B, bright
    data[:, 15:25, 25:35] = 800.0  # bar C, slightly dimmer
    rng = np.random.default_rng(0)
    data += rng.normal(0.0, 3.0, data.shape).astype(np.float32)
    return Volume(data=data, spacing=(1.0, 1.0, 1.0), description="blobs")


@pytest.fixture(scope="session")
def qt_app():
    """An offscreen QApplication for the UI regression tests."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PyQt5.QtWidgets")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    return app


@pytest.fixture(scope="session")
def real_volumes() -> Path:
    """The real Volumes directory, or skip if it was not provided."""
    path = os.environ.get(REAL_DATA_ENV)
    if not path or not Path(path).is_dir():
        pytest.skip(f"set {REAL_DATA_ENV} to the real Volumes folder to run this")
    return Path(path)
