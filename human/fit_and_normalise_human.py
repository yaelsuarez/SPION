#!/usr/bin/env python3
"""I0 and rs for the human/mouse intestine samples, with the phantom normalisation.

Creates new files only. Nothing under DATA_MRI or under the existing Project II
tree is modified, moved or deleted.

METHODOLOGY, reproduced from the existing work rather than reinvented
---------------------------------------------------------------------
Fitting follows i0_rs/Volumes/Syringes exactly: every voxel's 26-point decay is
fitted twice, with Model A ``I0*exp(-rs*TE)`` and Model B ``I0*exp(-rs*TE)+B``,
and the model with the lower AIC is kept (ties to A). rs_selected, I0_selected
and B_selected are the parameters of the chosen model. Quality flags use the
established thresholds: flag_high_rs when rs > 0.15 1/ms, flag_low_information
when fewer than 4 informative echoes.

Normalisation follows Normalized_data_i0_rs/.../Pill1 and the label-light scheme
that superseded it, in two steps:
    I0_normalized = I0_selected / reference           (per-sample scale)
    I0_residual   = I0_normalized - f(rs)             (rs-residual subtraction)
f(rs) is the FROZEN degree-2 polynomial from the phantom study, read from the
stored metadata and applied unchanged - it is not refitted here.

TWO ASSUMPTIONS, both forced by the data and both flagged in the outputs
-----------------------------------------------------------------------
1. Reference. The phantom rule is "median I0_selected of the sample's own
   0-SPION segmentation (quality_ok voxels)". These samples contain no SPION at
   all, so there is no 0-SPION segmentation and Pill1's own factor - transferred
   from its 0.2 segmentation - cannot be reproduced either. The closest faithful
   analogue is used: the median I0_selected over all of that sample's segmented
   (tissue) voxels, quality_ok only, one factor per sample. This is recorded in
   every output as `reference_rule`.
2. Masks. The NIfTI segmentations are stored transposed relative to the DICOM
   stack and are read with np.transpose(mask, (2, 1, 0)). This was verified, not
   assumed: with that orientation and a Model-A fit, the mean I0 and mean rs of
   each segmentation reproduce the values in the sample's own results.xlsx to
   the decimal.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT, HUMAN_SOURCE, bootstrap  # noqa: E402
bootstrap()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DATA = HUMAN_SOURCE
PROJECT = DATA_ROOT
FRS_SOURCE = PROJECT / "i0_rs" / "Volume_for_testing" / "Pill2_normalized" / "metadata.json"
OUT = PROJECT / "human"
SAMPLES = ["MSMECecam", "MSMEIleumBottom", "MSMEIleumTop", "MSMEfromBox"]
HIGH_RS = 0.15                 # 1/ms, the established flag threshold
MIN_INFORMATIVE = 4
SAMPLE_FRACTION = 0.10
SEED = 42
WORKERS = 9


def load_sample(sample, get_echotimes, load_volume):
    """Echo times, the 4-D stack and every mask, masks already re-oriented."""
    import nibabel as nib

    echoes = get_echotimes(DATA / sample / "DICOM" / "T2")
    te = np.array([e.value for e in echoes], float)
    stack = np.stack([load_volume(e.path).data.astype(np.float32) for e in echoes])
    shape = stack.shape[1:]
    masks = {}
    for path in sorted((DATA / sample / "DICOM" / "Segmentations").glob("*.nii.gz"),
                       key=lambda p: int(p.name.split(".")[0])):
        raw = np.asanyarray(nib.load(str(path)).dataobj)
        mask = np.transpose(raw, (2, 1, 0))       # verified against results.xlsx
        assert mask.shape == shape, f"{sample}/{path.name}: {mask.shape} vs {shape}"
        masks[path.name.split(".")[0]] = mask > 0
    return te, stack, masks, shape


def main() -> int:
    from echoviewer.dataset import get_available_echotimes, load_volume_from_folder
    from echoviewer.fitting import fit_signals

    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("human_voxels_10pct.xlsx", "human_segmentation_means.xlsx"):
        assert not (OUT / name).exists(), f"{name} already exists; refusing to overwrite"

    frs = json.loads(FRS_SOURCE.read_text())["normalization"]["f_rs"]
    f = np.poly1d(frs["coefficients_high_to_low"])     # frozen, not refitted
    print("f(rs) reused unchanged from the phantom study: "
          + " ".join(f"{c:+.6g}" for c in frs["coefficients_high_to_low"]))

    frames, published_check = [], []
    for sample in SAMPLES:
        te, stack, masks, shape = load_sample(sample, get_available_echotimes,
                                              load_volume_from_folder)
        print(f"\n{sample}: volume {shape}, {len(te)} echoes, "
              f"{len(masks)} segmentations")
        results = pd.read_excel(DATA / sample / "results.xlsx")
        results["concentration"] = results.concentration.astype(str)

        for seg, mask in masks.items():
            index = np.argwhere(mask)
            signals = stack[:, index[:, 0], index[:, 1], index[:, 2]].T.astype(np.float32)
            r = fit_signals(te, signals, workers=WORKERS)
            chose_a = r["model_selected"] == 1
            frame = pd.DataFrame({
                "sample": sample,
                "segmentation": seg,
                "sample_segmentation": f"{sample}_{seg}",
                "z": index[:, 0], "y": index[:, 1], "x": index[:, 2],
                "rs_selected": np.where(chose_a, r["rs_noB"], r["rs_B"]),
                "I0_selected": np.where(chose_a, r["I0_noB"], r["I0_B"]),
                "B_selected": np.where(chose_a, 0.0, r["B"]),
                "model_selected": r["model_selected"],
                "n_informative": r["n_points"],
                "fit_success": r["fit_success"],
                # Model A alone, kept purely to reproduce the published table
                "I0_modelA": r["I0_noB"], "rs_modelA": r["rs_noB"],
                "success_modelA": r["success_noB"],
            })
            frame["flag_high_rs"] = (frame.rs_selected > HIGH_RS).astype(np.uint8)
            frame["flag_low_information"] = (frame.n_informative < MIN_INFORMATIVE).astype(np.uint8)
            frame["quality_ok"] = ((frame.fit_success > 0) & (frame.flag_high_rs == 0)
                                   & (frame.flag_low_information == 0)).astype(np.uint8)
            frames.append(frame)

            a = frame[frame.success_modelA > 0]
            row = results[results.concentration == seg]
            published_check.append({
                "sample": sample, "segmentation": seg, "n_voxels": len(frame),
                "i0_modelA_mean_here": a.I0_modelA.mean(),
                "i0_published": float(row.i0.iloc[0]) if len(row) else np.nan,
                "rs_modelA_mean_here_per_s": a.rs_modelA.mean() * 1000,
                "rs_published_per_s": float(row.rs.iloc[0]) if len(row) else np.nan})
            print(f"   seg {seg:>2s}: {len(frame):>6,} voxels  "
                  f"rs_selected median {np.median(frame.rs_selected):.5f} 1/ms  "
                  f"I0_selected median {np.median(frame.I0_selected):9.1f}")
        del stack

    voxels = pd.concat(frames, ignore_index=True); del frames

    # ---- step 1: per-sample reference, tissue median --------------------------
    references = {}
    for sample, block in voxels.groupby("sample"):
        good = block[block.quality_ok == 1]
        references[sample] = float(np.median(good.I0_selected))
    voxels["normalization_factor"] = voxels["sample"].map(references)
    voxels["I0_normalized"] = voxels.I0_selected / voxels.normalization_factor
    # ---- step 2: rs-residual subtraction, frozen f(rs) ------------------------
    voxels["I0_residual"] = voxels.I0_normalized - f(voxels.rs_selected)
    print("\nper-sample reference (median I0_selected of that sample's tissue voxels, "
          "quality_ok):")
    for sample, value in references.items():
        print(f"   {sample:16s} {value:10.2f}")

    check = pd.DataFrame(published_check)
    check["i0_difference"] = (check.i0_modelA_mean_here - check.i0_published).abs()
    check["rs_difference"] = (check.rs_modelA_mean_here_per_s - check.rs_published_per_s).abs()
    print(f"\nreproduction of the published results.xlsx values: "
          f"max |I0 difference| {check.i0_difference.max():.3f}, "
          f"max |rs difference| {check.rs_difference.max():.4f} 1/s")

    # ---- outputs -------------------------------------------------------------
    voxels.to_csv(OUT / "human_voxels_all.csv.gz", index=False, compression="gzip")

    rng = np.random.default_rng(SEED)
    subset = (voxels.groupby("sample_segmentation", group_keys=False)
              .apply(lambda g: g.sample(frac=SAMPLE_FRACTION,
                                        random_state=SEED))).sort_index()
    ten = subset.rename(columns={"I0_selected": "i0",
                                 "I0_residual": "i0_after_second_normalization",
                                 "rs_selected": "relaxation_rate"})
    ten = ten[["sample_segmentation", "sample", "segmentation", "z", "y", "x",
               "i0", "i0_after_second_normalization", "relaxation_rate",
               "I0_normalized", "normalization_factor", "model_selected",
               "flag_high_rs", "flag_low_information", "quality_ok"]]

    means = (voxels.groupby(["sample", "segmentation", "sample_segmentation"])
             .agg(n_voxels=("I0_selected", "size"),
                  mean_i0=("I0_selected", "mean"),
                  mean_i0_after_second_normalization=("I0_residual", "mean"),
                  mean_rs_selected=("rs_selected", "mean"),
                  mean_I0_normalized=("I0_normalized", "mean"),
                  median_i0=("I0_selected", "median"),
                  median_rs_selected=("rs_selected", "median"),
                  std_rs_selected=("rs_selected", "std"),
                  normalization_factor=("normalization_factor", "first"),
                  n_quality_ok=("quality_ok", "sum"))
             .reset_index())
    means["mean_rs_selected_per_s"] = means.mean_rs_selected * 1000

    notes = pd.DataFrame([
        ("Fitting", "two models per voxel, AIC-selected; identical to i0_rs/Volumes/Syringes"),
        ("Model A", "I0 * exp(-rs * TE)"),
        ("Model B", "I0 * exp(-rs * TE) + B"),
        ("rs units", "1/ms (multiply by 1000 for 1/s, as used in the source results.xlsx)"),
        ("i0", "I0_selected, the amplitude of the AIC-selected model"),
        ("Normalisation step 1", "I0_normalized = I0_selected / reference"),
        ("Normalisation step 2", "i0_after_second_normalization = I0_normalized - f(rs)"),
        ("f(rs)", "frozen degree-2 polynomial from the phantom study, applied unchanged: "
                  + " ".join(f"{c:+.10g}" for c in frs["coefficients_high_to_low"])),
        ("ASSUMPTION reference_rule",
         "no 0-SPION segmentation exists in these data, so the reference is the median "
         "I0_selected over all of that sample's segmented tissue voxels (quality_ok), "
         "one factor per sample"),
        ("VERIFIED mask orientation",
         "NIfTI masks transposed (2,1,0) onto the DICOM stack; with that orientation a "
         "Model-A fit reproduces each sample's own results.xlsx to the decimal"),
        ("Voxels fitted", "every voxel inside each segmentation mask, no Otsu filter"),
        ("Subsample", f"{SAMPLE_FRACTION:.0%} per segmentation, seed {SEED}"),
        ("Sources", f"{DATA}"),
        ("Created", time.strftime("%Y-%m-%d %H:%M:%S")),
    ], columns=["item", "detail"])

    with pd.ExcelWriter(OUT / "human_voxels_10pct.xlsx", engine="openpyxl") as writer:
        ten.to_excel(writer, sheet_name="voxels_10pct", index=False)
        notes.to_excel(writer, sheet_name="_README", index=False)
    with pd.ExcelWriter(OUT / "human_segmentation_means.xlsx", engine="openpyxl") as writer:
        means.to_excel(writer, sheet_name="segmentation_means", index=False)
        check.round(4).to_excel(writer, sheet_name="published_reproduction", index=False)
        notes.to_excel(writer, sheet_name="_README", index=False)

    (OUT / "provenance.json").write_text(json.dumps({
        "source_data": str(DATA), "samples": SAMPLES,
        "n_segmentations": int(voxels.sample_segmentation.nunique()),
        "n_voxels": int(len(voxels)),
        "fitting": "AIC-selected two-model fit, as in i0_rs/Volumes/Syringes",
        "normalization": {"step_1": "I0_selected / reference",
                          "step_2": "I0_normalized - f(rs)",
                          "f_rs": frs, "references": references,
                          "reference_rule": "median I0_selected of that sample's segmented "
                                            "tissue voxels (quality_ok); no 0-SPION exists"},
        "mask_orientation": "np.transpose(nifti, (2,1,0)), verified against results.xlsx",
        "existing_files_modified": False,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2, default=float))

    print(f"\n{len(voxels):,} voxels across {voxels.sample_segmentation.nunique()} "
          f"segmentations; 10% subset = {len(ten):,} rows")
    print(f"saved to {OUT}")
    print(f"elapsed {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
