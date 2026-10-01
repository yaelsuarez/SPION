#!/usr/bin/env python3
"""Pill2 probability volumes and a single all-class label volume, per classifier.

Mirrors the layout already present for the GMM, for the nine other models. No
model is reloaded, retrained or refitted: the stored per-voxel probabilities are
read and reshaped. Only new files are written; each classifier's folder must not
already exist.

Per classifier:
  npy/probability_<class>.npy   original probabilities, unthresholded
  npy/valid_mask.npy
  all_clases/                   one volume holding every class as a label value
      labels_thresholded.npy    winning class where its probability reaches 0.5
      labels_argmax_unthresholded.npy
      max_probability.npy       original winning-class probability, unchanged
      dicom/                    that label volume, readable by dicom_viewer
      all_classes_overview.svg  labels at alpha 0.5 over the 8 ms echo, sagittal
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
SRC = MRI / "Classifiers" / "Classifiers"
OUT = MRI / "Results" / "Classifiers_probabilities_volumes"
CLASSES = ["0", "0.05", "0.1", "0.2", "0.25", "0.3", "tissue"]
MODELS = ["logistic_regression", "random_forest", "gradient_boosting", "rbf_svm", "knn",
          "lda", "qda", "hierarchical", "ordinal"]          # GMM already exists
SHAPE = (128, 330, 128)
SPACING = (0.32, 0.32, 0.32)
THRESHOLD = 0.5
ECHO_MS = 8.0
LABEL_MAP = {i + 1: c for i, c in enumerate(CLASSES)}
PALETTE = ["#000000", "#4c72b0", "#dd8452", "#55a868", "#c44e52",
           "#8172b3", "#937860", "#da8bc3"]


def overview(labels, background, lo, hi, path, title):
    """Label map at alpha 0.5 over the first-echo volume, sagittal slices."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from echoviewer.diagnostics import _figure

    present = np.flatnonzero((labels > 0).sum(axis=(0, 1)) > 0)
    slices = list(range(int(present.min()), int(present.max()) + 1,
                        max(1, (int(present.max()) - int(present.min())) // 7)))[:8]
    fig = _figure(figsize=(3.1 * len(slices) + 3.4, 4.6))
    axes = fig.subplots(1, len(slices), squeeze=False)[0]
    rgb = {i: tuple(int(PALETTE[i][j:j + 2], 16) / 255 for j in (1, 3, 5))
           for i in LABEL_MAP}
    for ax, x in zip(axes, slices):
        base = np.flipud(background[:, :, x])
        ax.imshow(base, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        plane = np.flipud(labels[:, :, x])
        layer = np.zeros(plane.shape + (4,), np.float32)
        for value, colour in rgb.items():
            m = plane == value
            if m.any():
                layer[m, 0], layer[m, 1], layer[m, 2], layer[m, 3] = (*colour, 0.5)
        ax.imshow(layer, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"sagittal x={x}", fontsize=9)
    fig.legend(handles=[Patch(facecolor=PALETTE[i], alpha=.5, edgecolor="black",
                              label=f"{i} = {c}") for i, c in LABEL_MAP.items()],
               loc="center right", fontsize=9, title="class label")
    fig.suptitle(title, fontsize=13)
    fig.subplots_adjust(right=0.88)
    for suffix in (".svg", ".png"):
        fig.savefig(path.with_suffix(suffix), dpi=130, bbox_inches="tight",
                    facecolor="white")
    fig.clf()


def main() -> int:
    from data_treatment.gmm_probability_volume import ordered_source_datasets
    from echoviewer.dataset import get_available_echotimes, load_volume_from_folder
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    started = time.time()
    for name in MODELS:
        assert not (OUT / name).exists(), f"{OUT / name} already exists; refusing to touch it"

    datasets, _ = ordered_source_datasets()
    assert (len(datasets), datasets[0][1].Rows, datasets[0][1].Columns) == SHAPE

    echo = min(get_available_echotimes(MRI / "Volumes" / "Pill2"),
               key=lambda e: abs(e.value - ECHO_MS))
    background = load_volume_from_folder(echo.path).data.astype(np.float32)
    lo, hi = np.percentile(background, [1, 99.5])
    print(f"background: Pill2 TE {echo.value:g} ms, window {lo:.0f}-{hi:.0f}\n")

    summary = []
    for name in MODELS:
        root = OUT / name
        (root / "npy").mkdir(parents=True)
        allc = root / "all_clases"
        (allc / "dicom").mkdir(parents=True)

        d = pd.read_csv(SRC / name / "pill2_fullvolume_predictions.csv.gz")
        z = d.z.to_numpy(np.intp); y = d.y.to_numpy(np.intp); x = d.x.to_numpy(np.intp)
        proba = d[[f"p_{c}" for c in CLASSES]].to_numpy(np.float32)
        valid = np.zeros(SHAPE, bool); valid[z, y, x] = True
        np.save(root / "npy" / "valid_mask.npy", valid)
        for i, c in enumerate(CLASSES):          # ORIGINAL values, unthresholded
            vol = np.zeros(SHAPE, np.float32)
            vol[z, y, x] = proba[:, i]
            np.save(root / "npy" / f"probability_{c}.npy", vol)
            del vol

        winner = proba.argmax(1); best = proba.max(1)
        labels_all = np.zeros(SHAPE, np.uint8); labels_all[z, y, x] = winner + 1
        max_prob = np.zeros(SHAPE, np.float32); max_prob[z, y, x] = best
        keep = best >= THRESHOLD
        labels = np.zeros(SHAPE, np.uint8)
        labels[z[keep], y[keep], x[keep]] = winner[keep] + 1
        np.save(allc / "labels_thresholded.npy", labels)
        np.save(allc / "labels_argmax_unthresholded.npy", labels_all)
        np.save(allc / "max_probability.npy", max_prob)
        np.save(allc / "valid_mask.npy", valid)

        n_valid = int(valid.sum())
        rows = [{"label_value": i, "class": c,
                 "voxels_argmax_no_threshold": int((labels_all == i).sum()),
                 "voxels_retained_at_0.5": int((labels == i).sum()),
                 "percent_of_fitted": 100 * int((labels == i).sum()) / n_valid}
                for i, c in LABEL_MAP.items()]
        rows.append({"label_value": 0, "class": "background / below threshold",
                     "voxels_argmax_no_threshold": 0,
                     "voxels_retained_at_0.5": n_valid - int((labels > 0).sum()),
                     "percent_of_fitted": 100 * (n_valid - int((labels > 0).sum())) / n_valid})
        pd.DataFrame(rows).round(4).to_csv(allc / "label_report.csv", index=False)

        series_uid = generate_uid()
        frame_uid = getattr(datasets[0][1], "FrameOfReferenceUID", None) or generate_uid()
        legend = "; ".join(f"{i}={c}" for i, c in LABEL_MAP.items())
        for k, (_, source) in enumerate(datasets):
            ds = copy.deepcopy(source)            # originals are never written to
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
            ds.SeriesDescription = f"{name} all classes (label map, thresh 0.5)"
            ds.ImageComments = (f"Voxel value = class label. 0 = background or max "
                                f"probability below {THRESHOLD:g}. {legend}")
            ds.Rows, ds.Columns = plane.shape
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated = ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = 0
            ds.RescaleSlope, ds.RescaleIntercept = 1.0, 0.0
            ds.RescaleType = "LABEL"
            ds.WindowCenter = len(CLASSES) / 2.0
            ds.WindowWidth = float(len(CLASSES) + 1)
            for tag in ("SmallestImagePixelValue", "LargestImagePixelValue"):
                if tag in ds:
                    del ds[tag]
            ds.PixelData = plane.tobytes()
            ds.save_as(allc / "dicom" / f"all_classes_{k:03d}.dcm",
                       enforce_file_format=True)

        overview(labels, background, lo, hi, allc / "all_classes_overview",
                 f"{name} — all-class label map at alpha 0.5 over the "
                 f"{echo.value:g} ms echo, sagittal")

        (root / "README.json").write_text(json.dumps({
            "created_from": f"Classifiers/Classifiers/{name}/pill2_fullvolume_predictions.csv.gz",
            "model": f"existing fitted {name}; not reloaded, retrained or refitted",
            "probabilities": "used exactly as stored",
            "shape_zyx": list(SHAPE), "voxel_spacing_mm": list(SPACING),
            "npy": "ORIGINAL probabilities, unthresholded, float32, 0 outside the fitted volume",
            "all_clases": ("one volume holding every class as a label value; the DICOM and "
                           f"the overview apply the {THRESHOLD:g} threshold for visualisation "
                           "only, the npy probabilities are untouched"),
            "label_map": {str(k): v for k, v in LABEL_MAP.items()},
            "n_valid_voxels": n_valid,
            "voxels_retained_at_threshold": int((labels > 0).sum()),
            "originals_modified": False,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, indent=2, default=str))

        kept = int((labels > 0).sum())
        summary.append({"classifier": name, "shape": "x".join(map(str, SHAPE)),
                        "valid_voxels": n_valid, "retained_at_0.5": kept,
                        "percent_retained": round(100 * kept / n_valid, 2)})
        print(f"  {name:20s} shape {SHAPE}  valid {n_valid:,}  "
              f"retained {kept:,} ({100*kept/n_valid:.2f}%)  [{time.time()-started:.0f}s]")
        del d, proba, labels, labels_all, max_prob, valid

    frame = pd.DataFrame(summary)
    frame.to_csv(OUT / "retention_summary_nine_models.csv", index=False)
    print("\n" + frame.to_string(index=False))
    print(f"\nsaved to {OUT}\nelapsed {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
