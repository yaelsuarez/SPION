#!/usr/bin/env python3
"""Per-class 3D IoU on Pill2 at a 0.7 confidence threshold, saved probabilities only.

Voxels whose highest class probability is below the threshold are discarded and
belong to no class; IoU is then computed over the retained voxels alone, so both
the prediction and the ground truth are restricted to the same voxel set.

Nothing is refitted. Writes into a new IoU/ directory and touches nothing that
exists.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MRI = DATA_ROOT
OUT = MRI / "Classifiers"
IOU_DIR = OUT / "IoU"
P2_COMPLETE = MRI / "i0_rs" / "Volume_for_testing" / "Pill2_segmentation_complete" / "voxels_complete.csv.gz"
ORDER = ["logistic_regression", "random_forest", "gradient_boosting", "rbf_svm", "knn",
         "lda", "qda", "gmm", "hierarchical", "ordinal"]
CLASSES = ["0", "0.05", "0.1", "0.2", "0.25", "0.3", "tissue"]
DEFAULT_THRESHOLD = 0.7


def main(threshold: float = DEFAULT_THRESHOLD) -> int:
    from echoviewer.diagnostics import _figure, _save

    # The default run writes to IoU/; any other threshold gets its own
    # subdirectory so earlier results are never touched.
    global IOU_DIR
    THRESHOLD = threshold
    if threshold != DEFAULT_THRESHOLD:
        # threshold 0 keeps every voxel: the argmax label with no confidence bar
        IOU_DIR = IOU_DIR / ("all_probability" if threshold <= 0
                             else f"{threshold:g}_probability")
    IOU_DIR.mkdir(parents=True, exist_ok=True)
    print(f"threshold {THRESHOLD} -> {IOU_DIR}")

    # ground-truth segmentation masks, for the voxel-wise alignment check
    truth_src = pd.read_csv(P2_COMPLETE, usecols=["z", "y", "x", "material"],
                            dtype={"material": str})
    truth_src = truth_src[truth_src.material.isin(CLASSES)].reset_index(drop=True)
    truth_key = set(map(tuple, truth_src[["z", "y", "x"]].to_numpy()))
    print(f"segmentation masks: {len(truth_src):,} labelled Pill2 voxels")

    rows, detail, alignment = [], [], {}
    for name in ORDER:
        path = OUT / name / "pill2_segmented_predictions.csv.gz"
        if not path.is_file():
            continue
        d = pd.read_csv(path, dtype={"material": str})
        cols = [f"p_{c}" for c in CLASSES]
        proba = d[cols].to_numpy(np.float64)

        # voxel-wise alignment: same coordinates, and the carried label matches
        # the segmentation mask row for row
        same_coords = set(map(tuple, d[["z", "y", "x"]].to_numpy())) == truth_key
        same_labels = bool((d.material.to_numpy() == truth_src.material.to_numpy()).all()) \
            if len(d) == len(truth_src) else False
        alignment[name] = {"n_rows": int(len(d)), "coordinates_match_masks": same_coords,
                           "labels_match_masks": same_labels}
        assert same_coords and same_labels, f"{name} is not voxel-aligned with the masks"

        best = proba.max(1)
        keep = best >= THRESHOLD
        pred = np.asarray(CLASSES)[proba.argmax(1)]
        truth = d.material.to_numpy()
        n_keep = int(keep.sum())
        coverage = 100.0 * n_keep / len(d)
        pred_k, truth_k = pred[keep], truth[keep]

        ious = {}
        for c in CLASSES:
            p = pred_k == c
            t = truth_k == c
            inter = int((p & t).sum())
            union = int((p | t).sum())
            iou = np.nan if union == 0 else inter / union
            ious[c] = iou
            rows.append({"classifier": name, "class": c,
                         "IoU": iou, "support_voxels": int(t.sum()),
                         "coverage": round(coverage, 4)})
            detail.append({"classifier": name, "class": c, "intersection": inter,
                           "union": union, "n_predicted_retained": int(p.sum()),
                           "n_truth_retained": int(t.sum()),
                           "n_truth_all": int((truth == c).sum()),
                           "IoU": iou,
                           "status": ("undefined: no retained voxel is this class in either "
                                      "prediction or ground truth" if union == 0
                                      else "ok")})
        defined = [c for c in CLASSES if np.isfinite(ious[c])]
        macro = float(np.mean([ious[c] for c in defined])) if defined else np.nan
        rows.append({"classifier": name, "class": "macro_mean", "IoU": macro,
                     "support_voxels": n_keep, "coverage": round(coverage, 4)})
        print(f"  {name:20s} retained {n_keep:>7,} ({coverage:5.1f}%)  "
              f"macro IoU {macro:.4f} over {len(defined)} classes")
        del d, proba

    summary = pd.DataFrame(rows)[["classifier", "class", "IoU", "support_voxels", "coverage"]]
    target = IOU_DIR / "iou_summary.csv"
    assert not target.is_file(), "iou_summary.csv already exists; refusing to overwrite"
    summary.round(6).to_csv(target, index=False)
    pd.DataFrame(detail).round(6).to_csv(IOU_DIR / "iou_detail.csv", index=False)

    wide = summary[summary["class"] != "macro_mean"].pivot(
        index="classifier", columns="class", values="IoU").reindex(columns=CLASSES)
    wide["macro_mean"] = summary[summary["class"] == "macro_mean"].set_index("classifier").IoU
    wide["coverage_pct"] = summary.groupby("classifier").coverage.first()
    wide = wide.reindex([n for n in ORDER if n in wide.index])
    wide.round(4).to_csv(IOU_DIR / "iou_matrix.csv")

    fig = _figure(figsize=(13, 6))
    ax = fig.subplots()
    plot = wide[CLASSES]
    idx = np.arange(len(plot))
    width = 0.8 / len(CLASSES)
    for i, c in enumerate(CLASSES):
        vals = plot[c].to_numpy(np.float64)
        ax.bar(idx + i * width - 0.4, np.nan_to_num(vals), width=width, label=c)
    ax.plot(idx, wide.macro_mean.to_numpy(), "ko--", lw=1.2, ms=5, label="macro mean")
    ax.set_xticks(idx); ax.set_xticklabels(plot.index, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("IoU"); ax.set_ylim(0, 1); ax.grid(alpha=.25, axis="y")
    ax.legend(fontsize=8, ncol=4)
    ax.set_title(f"Pill2 per-class 3D IoU, voxels with max probability ≥ {THRESHOLD} "
                 "(bars at 0 may be undefined — see iou_detail.csv)")
    _save(fig, IOU_DIR / "iou_by_class.png", [])

    (IOU_DIR / "notes.json").write_text(json.dumps({
        "threshold": THRESHOLD,
        "rule": ("hard label = argmax class, kept only where max probability >= threshold; "
                 "discarded voxels are assigned to no class and are excluded from both the "
                 "prediction and the ground truth before IoU"),
        "class_order": CLASSES,
        "support_voxels": "retained ground-truth voxels of that class",
        "coverage": "100 * retained / total labelled Pill2 voxels (same for every class row)",
        "macro_mean_row": ("class='macro_mean'; IoU averaged over classes with a defined IoU, "
                           "support_voxels holds the retained voxel count"),
        "zero_and_undefined": ("IoU = 0 means the class was predicted or present among "
                               "retained voxels but never both; IoU is blank (undefined) only "
                               "when the union is empty, i.e. the class appears in neither"),
        "no_refitting": "all numbers derive from saved pill2_segmented_predictions.csv.gz",
        "alignment_check": alignment,
    }, indent=2, default=float))
    pd.set_option("display.width", 220)
    print("\n" + wide.round(4).to_string())
    print(f"\nsaved to {IOU_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_THRESHOLD))
