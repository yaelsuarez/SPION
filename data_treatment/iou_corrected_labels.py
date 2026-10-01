#!/usr/bin/env python3
"""Pill2 IoU recomputed with Segmentation_4_0.25 relabelled as 0.2.

Only the ground-truth label changes. Predictions and probabilities are read from
the saved per-classifier files exactly as stored; no classifier is retrained and
no probability map is regenerated.

The correction is consequential rather than cosmetic: before it, Pill2 had no
0.2 ground truth at all, so IoU_0.2 was 0 for every classifier by construction
while 0.25 carried the truth. Afterwards the two swap roles, and the ~90% of
those voxels that were always predicted as 0.2 now count as correct.

Written as new files; nothing already in all_probability/ is touched.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MRI = DATA_ROOT
CLASSIFIERS = MRI / "Classifiers" / "Classifiers"
OUT = MRI / "Classifiers" / "Metrics" / "IoU" / "all_probability"
SEG_DIR = MRI / "Segmentations" / "Pill2" / "segmentations" / "Segmentation_4_0.25"
P2_COMPLETE = MRI / "i0_rs" / "Volume_for_testing" / "Pill2_segmentation_complete" / "voxels_complete.csv.gz"
CLASSES = ["0", "0.05", "0.1", "0.2", "0.25", "0.3", "tissue"]
RELABEL = {"0.25": "0.2"}
STEM = "iou_summary_corrected_0.25_as_0.2"


def main() -> int:
    from echoviewer.diagnostics import _figure, _save

    started = time.time()
    for suffix in (".csv", ".png"):
        assert not (OUT / f"{STEM}{suffix}").exists(), \
            f"{STEM}{suffix} already exists; refusing to overwrite"

    # confirm the relabelled voxels are exactly the ones in Segmentation_4_0.25
    complete = pd.read_csv(P2_COMPLETE, usecols=["material", "segmentation"],
                           dtype={"material": str, "segmentation": str})
    by_material = int((complete.material == "0.25").sum())
    by_folder = int((complete.segmentation == SEG_DIR.name).sum())
    print(f"voxels labelled 0.25            : {by_material:,}")
    print(f"voxels in {SEG_DIR.name}: {by_folder:,}")
    assert by_material == by_folder, "material '0.25' is not exactly Segmentation_4_0.25"
    print("-> the relabel touches exactly that segmentation\n")

    names = sorted(p.name for p in CLASSIFIERS.iterdir()
                   if (p / "pill2_segmented_predictions.csv.gz").is_file())
    rows, long_rows = [], []
    for name in names:
        d = pd.read_csv(CLASSIFIERS / name / "pill2_segmented_predictions.csv.gz",
                        dtype={"material": str, "predicted": str})
        truth = d.material.replace(RELABEL).to_numpy()      # ground truth only
        pred = d.predicted.to_numpy()                        # untouched
        entry = {"classifier": name}
        ious = {}
        for c in CLASSES:
            t, p = truth == c, pred == c
            inter, union = int((t & p).sum()), int((t | p).sum())
            iou = np.nan if union == 0 else inter / union
            ious[c] = iou
            entry[f"IoU_{c}"] = iou
            long_rows.append({"classifier": name, "class": c, "IoU": iou,
                              "intersection": inter, "union": union,
                              "n_truth": int(t.sum()), "n_predicted": int(p.sum()),
                              "status": "ok" if union else
                                        "undefined: class in neither truth nor prediction"})
        defined = [v for v in ious.values() if np.isfinite(v)]
        entry["IoU_macro_mean"] = float(np.mean(defined))
        entry["classes_scored"] = len(defined)
        entry["n_voxels"] = int(len(d))
        entry["coverage_pct"] = 100.0
        rows.append(entry)
        print(f"  {name:20s} macro {entry['IoU_macro_mean']:.4f}  "
              f"IoU_0.2 {ious['0.2']:.4f}  over {len(defined)} classes")
        del d

    summary = pd.DataFrame(rows)
    summary.round(6).to_csv(OUT / f"{STEM}.csv", index=False)
    pd.DataFrame(long_rows).round(6).to_csv(OUT / f"{STEM}_per_class_long.csv", index=False)

    # comparison against the uncorrected run already in this folder
    previous = pd.read_csv(OUT / "iou_matrix.csv", index_col=0)
    compare = pd.DataFrame({
        "classifier": summary.classifier,
        "macro_before": previous.reindex(summary.classifier)["macro_mean"].to_numpy(),
        "macro_after": summary.IoU_macro_mean.to_numpy(),
        "IoU_0.2_before": previous.reindex(summary.classifier)["0.2"].to_numpy(),
        "IoU_0.2_after": summary["IoU_0.2"].to_numpy(),
    })
    compare["macro_change"] = compare.macro_after - compare.macro_before
    compare.round(6).to_csv(OUT / f"{STEM}_vs_original.csv", index=False)

    order = summary.sort_values("IoU_macro_mean", ascending=False).classifier.tolist()
    plot = summary.set_index("classifier").reindex(order)
    fig = _figure(figsize=(14, 6.4))
    ax = fig.subplots()
    idx = np.arange(len(plot))
    width = 0.8 / len(CLASSES)
    for i, c in enumerate(CLASSES):
        values = plot[f"IoU_{c}"].to_numpy(np.float64)
        ax.bar(idx + i * width - 0.4, np.nan_to_num(values), width=width, label=c)
    ax.plot(idx, plot.IoU_macro_mean.to_numpy(), "ko--", lw=1.2, ms=5, label="macro mean")
    ax.set_xticks(idx); ax.set_xticklabels(plot.index, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("IoU"); ax.set_ylim(0, 1); ax.grid(alpha=.25, axis="y")
    ax.legend(fontsize=8, ncol=4, title="class")
    ax.set_title("Pill2 per-class IoU with Segmentation_4_0.25 relabelled as 0.2 — "
                 "all voxels, no confidence threshold")
    _save(fig, OUT / f"{STEM}.png", [])

    (OUT / f"{STEM}_README.json").write_text(json.dumps({
        "correction": "Pill2 ground-truth label 0.25 treated as 0.2",
        "affected_segmentation": str(SEG_DIR.relative_to(MRI)),
        "affected_voxels": by_material,
        "predictions": "unchanged; read from Classifiers/Classifiers/<name>/"
                       "pill2_segmented_predictions.csv.gz",
        "recomputed_probabilities": False, "retrained_classifiers": False,
        "threshold": "none; every labelled voxel retained, coverage 100%",
        "class_order": CLASSES,
        "undefined": "IoU blank where a class appears in neither truth nor prediction",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2, default=float))

    pd.set_option("display.width", 220)
    print("\n" + compare.round(4).to_string(index=False))
    print(f"\nsaved to {OUT}\nelapsed {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
