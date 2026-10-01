#!/usr/bin/env python3
"""Dense Pill2 class-probability volumes from the already-fitted GMM.

The GMM is not loaded or re-run: its per-voxel probabilities are already stored
in pill2_fullvolume_predictions.csv.gz, and those values are used exactly as
saved. Nothing existing is opened for writing; every output is a new file under
GMM_Pill2_Volume/.

Two representations are written, deliberately different:

* ``npy/`` holds the ORIGINAL probabilities, unthresholded, float32.
* ``dicom/`` holds a 0.5-thresholded copy for viewing only, encoded as uint16
  with RescaleSlope 1/65535 so dicom_viewer's loader turns the stored integer
  straight back into a probability in 0-1.

Slice order is taken from dicom_viewer's own rule - projection of
ImagePositionPatient onto the slice normal, ascending - by importing the
loader's helpers rather than re-deriving it, so the written series stacks the
same way the fitted volume did.
"""

import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT, bootstrap  # noqa: E402
bootstrap()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MRI = DATA_ROOT
GMM = MRI / "Classifiers" / "Classifiers" / "gmm"
SOURCE_DICOM = MRI / "Volumes" / "Pill2"
OUT = MRI / "Classifiers" / "GMM_Pill2_Volume"
CLASSES = ["0", "0.05", "0.1", "0.2", "0.25", "0.3", "tissue"]
SHAPE = (128, 330, 128)          # z (slice), y (row), x (column)
SPACING = (0.32, 0.32, 0.32)
THRESHOLD = 0.5
UINT16_MAX = 65535


def ordered_source_datasets():
    """Pill2 slice headers in the order dicom_viewer stacks them."""
    import pydicom
    from dicomview.loader import _slice_normal

    folder = sorted(p for p in SOURCE_DICOM.iterdir() if p.is_dir())[0]
    files = sorted(folder.glob("*.dcm"))
    datasets = [(p, pydicom.dcmread(str(p))) for p in files]
    normal = _slice_normal(getattr(datasets[0][1], "ImageOrientationPatient", None))

    def key(item):
        index, (path, ds) = item
        position = getattr(ds, "ImagePositionPatient", None)
        if position is not None and len(position) == 3:
            projection = float(np.dot([float(v) for v in position], normal))
        else:
            projection = float(getattr(ds, "InstanceNumber", index) or index)
        return projection, float(index)

    ordered = [pair for _, pair in sorted(enumerate(datasets), key=key)]
    return ordered, folder


def write_series(volume, label, datasets, out_dir, series_number):
    """One .dcm per slice, uint16 with a rescale back to probability."""
    import pydicom
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    out_dir.mkdir(parents=True, exist_ok=True)
    series_uid = generate_uid()
    frame_uid = getattr(datasets[0][1], "FrameOfReferenceUID", None) or generate_uid()
    for k, (path, source) in enumerate(datasets):
        ds = copy.deepcopy(source)                  # the original is never touched
        plane = np.clip(volume[k], 0.0, 1.0)
        stored = np.rint(plane * UINT16_MAX).astype(np.uint16)

        ds.file_meta = copy.deepcopy(source.file_meta)
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        ds.is_little_endian, ds.is_implicit_VR = True, False
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame_uid
        ds.SeriesNumber = series_number
        ds.InstanceNumber = k + 1
        ds.SeriesDescription = f"GMM p({label}) thresh {THRESHOLD:g}"
        ds.ImageComments = (f"GMM class probability for '{label}'. Stored value x "
                            f"RescaleSlope = probability in 0-1. Values below "
                            f"{THRESHOLD:g} set to 0 for visualisation.")
        ds.Rows, ds.Columns = stored.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.RescaleSlope = 1.0 / UINT16_MAX
        ds.RescaleIntercept = 0.0
        ds.RescaleType = "PROB"
        ds.WindowCenter = 0.5
        ds.WindowWidth = 1.0
        for tag in ("SmallestImagePixelValue", "LargestImagePixelValue"):
            if tag in ds:
                del ds[tag]
        ds.PixelData = stored.tobytes()
        ds.save_as(out_dir / f"{label.replace('.', 'p')}_{k:03d}.dcm",
                   enforce_file_format=True)
    return series_uid


def main() -> int:
    from echoviewer.diagnostics import _figure, _save

    started = time.time()
    # The folder may already exist holding only Finder's .DS_Store. Anything
    # else in it is a real result and this script must not write over it.
    existing = [p for p in OUT.rglob("*") if p.is_file() and p.name != ".DS_Store"] \
        if OUT.exists() else []
    assert not existing, f"{OUT} already contains {len(existing)} file(s); refusing to overwrite"
    (OUT / "npy").mkdir(parents=True, exist_ok=True)

    datasets, source_folder = ordered_source_datasets()
    rows, cols = datasets[0][1].Rows, datasets[0][1].Columns
    print(f"source geometry: {len(datasets)} slices of {rows} x {cols} "
          f"from {source_folder.name}")
    assert (len(datasets), rows, cols) == SHAPE, \
        f"source geometry {(len(datasets), rows, cols)} != fitted volume {SHAPE}"

    d = pd.read_csv(GMM / "pill2_fullvolume_predictions.csv.gz")
    z = d.z.to_numpy(np.intp); y = d.y.to_numpy(np.intp); x = d.x.to_numpy(np.intp)
    print(f"GMM probabilities: {len(d):,} fitted voxels (values used exactly as stored)")

    valid = np.zeros(SHAPE, bool)
    valid[z, y, x] = True
    np.save(OUT / "npy" / "valid_mask.npy", valid)

    report, volumes = [], {}
    for i, c in enumerate(CLASSES):
        vol = np.zeros(SHAPE, np.float32)
        vol[z, y, x] = d[f"p_{c}"].to_numpy(np.float32)
        np.save(OUT / "npy" / f"probability_{c}.npy", vol)      # original values
        shown = np.where(vol >= THRESHOLD, vol, 0.0).astype(np.float32)
        volumes[c] = shown
        kept = int((shown > 0).sum())
        report.append({"class": c, "shape": "x".join(map(str, SHAPE)),
                       "matches_pill2_volume": True,
                       "fitted_voxels": int(valid.sum()),
                       "voxels_retained_at_0.5": kept,
                       "percent_of_fitted": 100 * kept / int(valid.sum()),
                       "max_probability": float(vol.max()),
                       "mean_probability_over_fitted": float(vol[valid].mean())})
        uid = write_series(shown, c, datasets, OUT / "dicom" / f"class_{c}", 900 + i)
        report[-1]["series_instance_uid"] = uid
        print(f"  {c:7s} retained {kept:>7,} voxels "
              f"({100*kept/int(valid.sum()):5.2f}% of fitted)  -> dicom/class_{c}/")

    summary = pd.DataFrame(report)
    summary.round(6).to_csv(OUT / "volume_report.csv", index=False)

    # ---- visualisation: class label + probability scale --------------------
    slices = list(range(8, SHAPE[0] - 4, 12))
    fig = _figure(figsize=(1.5 * len(slices) + 3, 1.6 * len(CLASSES) + 2))
    axes = fig.subplots(len(CLASSES), len(slices), squeeze=False)
    im = None
    for r, c in enumerate(CLASSES):
        for j, k in enumerate(slices):
            ax = axes[r][j]
            plane = np.where(valid[k], volumes[c][k], np.nan)
            im = ax.imshow(plane, cmap="inferno", vmin=0, vmax=1, interpolation="nearest")
            ax.set_facecolor("black")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"slice {k}", fontsize=7)
            if j == 0:
                ax.set_ylabel(f"class {c}", fontsize=9)
    bar = fig.colorbar(im, cax=fig.add_axes([0.93, 0.08, 0.011, 0.84]))
    bar.set_label("GMM class probability (0–1); values < 0.5 set to 0 for display",
                  fontsize=9)
    fig.suptitle("Pill2 GMM class-probability volumes — 128 x 330 x 128, "
                 "0.32 mm isotropic", fontsize=13)
    _save(fig, OUT / "probability_volumes_overview.png", [])

    (OUT / "README.json").write_text(json.dumps({
        "created_from": str((GMM / "pill2_fullvolume_predictions.csv.gz").relative_to(MRI)),
        "model": "existing fitted GMM; not reloaded, retrained or refitted",
        "probabilities": "used exactly as stored",
        "shape_zyx": list(SHAPE), "voxel_spacing_mm": list(SPACING),
        "npy": "ORIGINAL probabilities, unthresholded, float32, 0 outside the fitted volume",
        "dicom": (f"visualisation copy with probabilities < {THRESHOLD} set to 0; uint16 with "
                  f"RescaleSlope 1/{UINT16_MAX}, so dicom_viewer shows probability directly "
                  "in 0-1. Window centre 0.5, width 1.0. Class named in SeriesDescription."),
        "slice_order": ("ImagePositionPatient projected on the slice normal, ascending - "
                        "dicom_viewer's own rule, imported from its loader"),
        "source_headers": str(source_folder.relative_to(MRI)),
        "originals_modified": False,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2, default=str))

    print(f"\n{summary[['class', 'voxels_retained_at_0.5', 'percent_of_fitted']].to_string(index=False)}")
    print(f"\nelapsed {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
