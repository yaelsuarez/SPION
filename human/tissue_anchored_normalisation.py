#!/usr/bin/env python3
"""Tissue-anchored I0 normalisation, applied identically to every dataset.

Each sample's I0_selected is divided by the median I0_selected of that sample's
own TISSUE voxels, so human samples, Pill1 and Pill2 are all pinned at 1.0 on
tissue and their I0-rs clouds can be compared by shape.

This is the only rule applicable to all six datasets: tissue is the sole
material they share, since the human samples contain no SPION and no 0 or water
reference.

WHAT THIS LICENSES, and what it does not
----------------------------------------
The position of each tissue distribution is 1.0 BY CONSTRUCTION and carries no
information. What can be compared is the shape of the cloud - spread, skew, and
how I0 varies with rs - and, within the phantoms, where the SPION classes fall
relative to their own tissue. rs is untouched throughout and is comparable
without any normalisation at all.

Read-only on every existing store. New files only.
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
OUT = MRI / "human"
HUMAN = OUT / "human_voxels_all.csv.gz"
PILL1 = MRI / "Normalized_data_i0_rs" / "Segmentations_training" / "Pill1"
PILL2 = MRI / "i0_rs" / "Volume_for_testing" / "Pill2_segmentation_complete" / "voxels_complete.csv.gz"
SAMPLE_FRACTION = 0.10
SEED = 42
STEM = "tissue_anchored"


def load_everything():
    """All six datasets in one frame, with a `material` and an `is_tissue` flag."""
    frames = []

    human = pd.read_csv(HUMAN, usecols=["sample", "segmentation", "sample_segmentation",
                                        "z", "y", "x", "rs_selected", "I0_selected",
                                        "flag_high_rs", "flag_low_information",
                                        "quality_ok"])
    human["dataset"] = "human"
    human["material"] = "tissue"          # every human segmentation is an organ
    frames.append(human)

    for seg in sorted(p for p in PILL1.iterdir() if p.is_dir()):
        block = pd.read_csv(seg / "voxels.csv.gz",
                            usecols=["x", "y", "z", "rs_selected", "I0_selected",
                                     "flag_high_rs", "flag_low_information", "quality_ok"])
        block["dataset"] = "phantom"
        block["sample"] = "Pill1"
        block["segmentation"] = seg.name
        block["sample_segmentation"] = f"Pill1_{seg.name}"
        block["material"] = seg.name.split("_", 2)[2]      # folder name is authoritative
        frames.append(block)

    p2 = pd.read_csv(PILL2, dtype={"material": str})
    p2["material"] = p2.segmentation.str.split("_", n=2).str[2]
    p2["dataset"] = "phantom"
    p2["sample"] = "Pill2"
    p2["sample_segmentation"] = "Pill2_" + p2.segmentation
    frames.append(p2[["dataset", "sample", "segmentation", "sample_segmentation", "material",
                      "z", "y", "x", "rs_selected", "I0_selected",
                      "flag_high_rs", "flag_low_information", "quality_ok"]])

    voxels = pd.concat(frames, ignore_index=True)
    voxels["is_tissue"] = (voxels.material == "tissue").astype(np.uint8)
    return voxels


def main() -> int:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    targets = {"voxels": OUT / f"{STEM}_voxels_10pct.xlsx",
               "means": OUT / f"{STEM}_segmentation_means.xlsx",
               "all": OUT / f"{STEM}_voxels_all.csv.gz"}
    for path in targets.values():
        assert not path.exists(), f"{path.name} already exists; refusing to overwrite"

    voxels = load_everything()
    print(f"{len(voxels):,} voxels, {voxels.sample_segmentation.nunique()} segmentations, "
          f"{voxels['sample'].nunique()} samples")

    # ---- the anchor: each sample's own tissue median ------------------------
    anchors = {}
    for sample, block in voxels.groupby("sample"):
        tissue = block[(block.is_tissue == 1) & (block.quality_ok == 1)]
        assert len(tissue), f"{sample} has no quality_ok tissue voxels"
        anchors[sample] = float(np.median(tissue.I0_selected))
    voxels["tissue_anchor"] = voxels["sample"].map(anchors)
    voxels["i0_tissue_anchored"] = voxels.I0_selected / voxels.tissue_anchor

    print("\ntissue anchor = median I0_selected of that sample's own tissue (quality_ok):")
    for sample, value in sorted(anchors.items(), key=lambda kv: kv[1]):
        tissue = voxels[(voxels["sample"] == sample) & (voxels.is_tissue == 1)]
        print(f"   {sample:16s} {value:11.2f}   tissue median after anchoring "
              f"{np.median(tissue.i0_tissue_anchored):.4f}")

    # ---- outputs, mirroring the two earlier human files --------------------
    voxels.to_csv(targets["all"], index=False, compression="gzip", float_format="%.7g")

    subset = (voxels.groupby("sample_segmentation", group_keys=False)
              .apply(lambda g: g.sample(frac=SAMPLE_FRACTION, random_state=SEED)))
    ten = (subset.rename(columns={"I0_selected": "i0", "rs_selected": "relaxation_rate"})
           [["sample_segmentation", "dataset", "sample", "segmentation", "material",
             "is_tissue", "z", "y", "x", "i0", "i0_tissue_anchored", "relaxation_rate",
             "tissue_anchor", "flag_high_rs", "flag_low_information", "quality_ok"]]
           .sort_values(["dataset", "sample", "segmentation"]).reset_index(drop=True))

    means = (voxels.groupby(["dataset", "sample", "segmentation", "sample_segmentation",
                             "material", "is_tissue"])
             .agg(n_voxels=("I0_selected", "size"),
                  mean_i0=("I0_selected", "mean"),
                  mean_i0_tissue_anchored=("i0_tissue_anchored", "mean"),
                  mean_rs_selected=("rs_selected", "mean"),
                  median_i0=("I0_selected", "median"),
                  median_i0_tissue_anchored=("i0_tissue_anchored", "median"),
                  median_rs_selected=("rs_selected", "median"),
                  std_rs_selected=("rs_selected", "std"),
                  iqr_i0_tissue_anchored=("i0_tissue_anchored",
                                          lambda s: s.quantile(.75) - s.quantile(.25)),
                  tissue_anchor=("tissue_anchor", "first"),
                  n_quality_ok=("quality_ok", "sum"))
             .reset_index())
    means["mean_rs_selected_per_s"] = means.mean_rs_selected * 1000

    anchor_sheet = pd.DataFrame([
        {"sample": s, "dataset": voxels[voxels["sample"] == s].dataset.iloc[0],
         "tissue_anchor": v,
         "n_tissue_voxels": int(((voxels["sample"] == s) & (voxels.is_tissue == 1)).sum()),
         "tissue_median_after_anchoring": float(np.median(
             voxels[(voxels["sample"] == s) & (voxels.is_tissue == 1)].i0_tissue_anchored))}
        for s, v in anchors.items()])

    notes = pd.DataFrame([
        ("Normalisation", "i0_tissue_anchored = I0_selected / tissue_anchor"),
        ("tissue_anchor", "median I0_selected of that sample's own TISSUE voxels, "
                          "quality_ok only - one value per sample"),
        ("Why this rule", "tissue is the only material shared by all six datasets; the human "
                          "samples contain no SPION and no 0 or water reference"),
        ("READ THIS", "every tissue distribution sits at 1.0 BY CONSTRUCTION. Compare the "
                      "SHAPE of the I0-rs cloud - spread, skew, dependence on rs - never the "
                      "position of the tissue medians"),
        ("rs", "untouched; rs_selected is comparable across datasets without normalisation. "
               "Units 1/ms; multiply by 1000 for 1/s"),
        ("Not applied here", "the f(rs) residual step is deliberately omitted: f(rs) was "
                             "fitted on phantom I0 normalised to a 0-SPION reference and "
                             "introduces an offset under tissue anchoring"),
        ("Pill2 labels", "material is taken from the segmentation folder name and is shown "
                         "as originally segmented; note 0.25 was later corrected to 0.2"),
        ("Subsample", f"{SAMPLE_FRACTION:.0%} per segmentation, seed {SEED}"),
        ("Sources", "MRI/human/human_voxels_all.csv.gz; "
                    "Normalized_data_i0_rs/Segmentations_training/Pill1; "
                    "i0_rs/Volume_for_testing/Pill2_segmentation_complete"),
        ("Existing files modified", "none"),
        ("Created", time.strftime("%Y-%m-%d %H:%M:%S")),
    ], columns=["item", "detail"])

    with pd.ExcelWriter(targets["voxels"], engine="openpyxl") as writer:
        ten.to_excel(writer, sheet_name="voxels_10pct", index=False)
        anchor_sheet.to_excel(writer, sheet_name="anchors", index=False)
        notes.to_excel(writer, sheet_name="_README", index=False)
    with pd.ExcelWriter(targets["means"], engine="openpyxl") as writer:
        means.to_excel(writer, sheet_name="segmentation_means", index=False)
        anchor_sheet.to_excel(writer, sheet_name="anchors", index=False)
        notes.to_excel(writer, sheet_name="_README", index=False)

    (OUT / f"{STEM}_provenance.json").write_text(json.dumps({
        "normalization": "I0_selected / median I0_selected of the sample's own tissue",
        "anchors": anchors,
        "datasets": sorted(voxels["sample"].unique().tolist()),
        "n_voxels": int(len(voxels)),
        "n_segmentations": int(voxels.sample_segmentation.nunique()),
        "subsample": {"fraction": SAMPLE_FRACTION, "seed": SEED, "rows": int(len(ten))},
        "f_rs_residual_applied": False,
        "existing_files_modified": False,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2, default=float))

    pd.set_option("display.width", 200)
    print("\nper-segmentation medians after anchoring")
    print(means[["sample", "segmentation", "material", "n_voxels",
                 "median_i0_tissue_anchored", "median_rs_selected"]]
          .round(4).to_string(index=False))
    print(f"\n10% subset: {len(ten):,} rows\nsaved to {OUT}\n"
          f"elapsed {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
