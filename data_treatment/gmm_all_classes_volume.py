#!/usr/bin/env python3
"""One Pill2 volume holding every GMM class, as a label map.

Each voxel carries the index of its winning class when that class's probability
reaches 0.5, and 0 where nothing does. This is the single-volume counterpart of
the seven per-class probability volumes: same geometry, same slice order, same
0.5 rule, but readable in one pass through dicom_viewer.

The GMM is not reloaded, retrained or refitted - its stored probabilities are
used exactly as saved. Only new files are written, under all_clases/.
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
BASE = MRI / "Classifiers" / "GMM_Pill2_Volume"
OUT = BASE / "all_clases"
CLASSES = ["0", "0.05", "0.1", "0.2", "0.25", "0.3", "tissue"]
SHAPE = (128, 330, 128)
SPACING = (0.32, 0.32, 0.32)
THRESHOLD = 0.5
#: 0 is reserved for background and for voxels no class claims at the threshold.
LABEL_MAP = {i + 1: c for i, c in enumerate(CLASSES)}


#: Categorical, because a grey ramp makes adjacent labels indistinguishable.
PALETTE = ["#000000", "#4c72b0", "#dd8452", "#55a868", "#c44e52",
           "#8172b3", "#937860", "#da8bc3"]


def build_overview(labels, max_prob, valid, path):
    """Label map with legend over the winning-class probability, PNG and SVG."""
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch
    from echoviewer.diagnostics import _figure

    cmap = ListedColormap(PALETTE)
    norm = BoundaryNorm(np.arange(-0.5, len(PALETTE) + 0.5), cmap.N)
    slices = list(range(8, SHAPE[0] - 4, 12))
    fig = _figure(figsize=(1.6 * len(slices) + 4, 7.6))
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1])

    axes = top.subplots(1, len(slices), squeeze=False)[0]
    for ax, k in zip(axes, slices):
        ax.imshow(labels[k], cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_facecolor("black"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"slice {k}", fontsize=7)
    axes[0].set_ylabel("class label", fontsize=9)
    top.legend(handles=[Patch(facecolor=PALETTE[0], label="0 = background / < 0.5")]
               + [Patch(facecolor=PALETTE[i], label=f"{i} = {c}")
                  for i, c in LABEL_MAP.items()],
               loc="center right", fontsize=8)
    top.suptitle("GMM all-class label map — one volume, 128 x 330 x 128, 0.32 mm isotropic",
                 fontsize=12)

    axes2 = bottom.subplots(1, len(slices), squeeze=False)[0]
    im = None
    for ax, k in zip(axes2, slices):
        im = ax.imshow(np.where(valid[k], max_prob[k], np.nan), cmap="inferno",
                       vmin=0, vmax=1, interpolation="nearest")
        ax.set_facecolor("black"); ax.set_xticks([]); ax.set_yticks([])
    axes2[0].set_ylabel("max probability", fontsize=9)
    bar = bottom.colorbar(im, ax=list(axes2), fraction=0.012, pad=0.01)
    bar.set_label("GMM winning-class probability (0–1)", fontsize=9)
    bottom.suptitle("Winning-class probability behind each label (original values, "
                    "not thresholded)", fontsize=11)

    written = []
    for suffix in (".svg", ".png"):          # SVG first; the PNG save clears the figure
        out = path.with_suffix(suffix)
        fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
        written.append(out)
    fig.clf()
    return written


def main() -> int:
    from data_treatment.gmm_probability_volume import ordered_source_datasets
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    started = time.time()
    existing = [p for p in OUT.rglob("*") if p.is_file() and p.name != ".DS_Store"] \
        if OUT.exists() else []
    assert not existing, f"{OUT} already contains {len(existing)} file(s); refusing to overwrite"
    OUT.mkdir(parents=True, exist_ok=True)

    datasets, source_folder = ordered_source_datasets()
    assert (len(datasets), datasets[0][1].Rows, datasets[0][1].Columns) == SHAPE, \
        "source geometry does not match the fitted volume"

    d = pd.read_csv(GMM / "pill2_fullvolume_predictions.csv.gz")
    z = d.z.to_numpy(np.intp); y = d.y.to_numpy(np.intp); x = d.x.to_numpy(np.intp)
    proba = d[[f"p_{c}" for c in CLASSES]].to_numpy(np.float32)
    winner = proba.argmax(1)
    best = proba.max(1)

    valid = np.zeros(SHAPE, bool); valid[z, y, x] = True
    labels_argmax = np.zeros(SHAPE, np.uint8)
    labels_argmax[z, y, x] = winner + 1
    max_prob = np.zeros(SHAPE, np.float32)
    max_prob[z, y, x] = best
    keep = best >= THRESHOLD
    labels = np.zeros(SHAPE, np.uint8)
    labels[z[keep], y[keep], x[keep]] = winner[keep] + 1

    np.save(OUT / "labels_thresholded.npy", labels)        # what the DICOM shows
    np.save(OUT / "labels_argmax_unthresholded.npy", labels_argmax)
    np.save(OUT / "max_probability.npy", max_prob)         # original values, untouched
    np.save(OUT / "valid_mask.npy", valid)

    rows = []
    for index, c in LABEL_MAP.items():
        n_all = int((labels_argmax == index).sum())
        n_keep = int((labels == index).sum())
        rows.append({"label_value": index, "class": c,
                     "voxels_argmax_no_threshold": n_all,
                     "voxels_retained_at_0.5": n_keep,
                     "percent_of_fitted": 100 * n_keep / int(valid.sum())})
    rows.append({"label_value": 0, "class": "background / below threshold",
                 "voxels_argmax_no_threshold": int(valid.sum() - (labels_argmax > 0).sum()),
                 "voxels_retained_at_0.5": int(valid.sum() - (labels > 0).sum()),
                 "percent_of_fitted": 100 * (valid.sum() - (labels > 0).sum()) / int(valid.sum())})
    report = pd.DataFrame(rows)
    report.round(4).to_csv(OUT / "label_report.csv", index=False)

    # ---- one DICOM series, label value stored directly --------------------
    dicom_dir = OUT / "dicom"
    dicom_dir.mkdir(exist_ok=True)
    series_uid = generate_uid()
    frame_uid = getattr(datasets[0][1], "FrameOfReferenceUID", None) or generate_uid()
    legend = "; ".join(f"{i}={c}" for i, c in LABEL_MAP.items())
    for k, (path, source) in enumerate(datasets):
        ds = copy.deepcopy(source)                 # originals are never written to
        plane = labels[k].astype(np.uint16)
        ds.file_meta = copy.deepcopy(source.file_meta)
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        ds.is_little_endian, ds.is_implicit_VR = True, False
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
        ds.SeriesInstanceUID = series_uid
        ds.FrameOfReferenceUID = frame_uid
        ds.SeriesNumber = 950
        ds.InstanceNumber = k + 1
        ds.SeriesDescription = "GMM all classes (label map, thresh 0.5)"
        ds.ImageComments = (f"Voxel value = class label. 0=background or max probability "
                            f"below {THRESHOLD:g}. {legend}")
        ds.Rows, ds.Columns = plane.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = 0.0
        ds.RescaleType = "LABEL"
        ds.WindowCenter = len(CLASSES) / 2.0
        ds.WindowWidth = float(len(CLASSES) + 1)
        for tag in ("SmallestImagePixelValue", "LargestImagePixelValue"):
            if tag in ds:
                del ds[tag]
        ds.PixelData = plane.tobytes()
        ds.save_as(dicom_dir / f"all_classes_{k:03d}.dcm", enforce_file_format=True)

    build_overview(labels, max_prob, valid, OUT / "all_classes_overview.png")

    (OUT / "README.json").write_text(json.dumps({
        "created_from": str((GMM / "pill2_fullvolume_predictions.csv.gz").relative_to(MRI)),
        "model": "existing fitted GMM; not reloaded, retrained or refitted",
        "content": "one volume holding every class as a label value",
        "label_map": {str(k): v for k, v in LABEL_MAP.items()},
        "label_0": f"background, or no class reaching {THRESHOLD}",
        "shape_zyx": list(SHAPE), "voxel_spacing_mm": list(SPACING),
        "dicom": ("single series, voxel value = label, RescaleSlope 1; window centre "
                  f"{len(CLASSES)/2} width {len(CLASSES)+1}"),
        "npy": {"labels_thresholded.npy": "what the DICOM shows",
                "labels_argmax_unthresholded.npy": "winning class everywhere, no threshold",
                "max_probability.npy": "original winning-class probability, unchanged"},
        "originals_modified": False,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2, default=str))

    print(report.to_string(index=False))
    print(f"\nsaved to {OUT}\nelapsed {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
