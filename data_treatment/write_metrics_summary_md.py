from pathlib import Path

M = DATA_ROOT / "Classifiers" / "Metrics"

text = """# Additional segmentation metrics on Pill2

Volume error, Brier score, MCC, HD95 and ASSD for all ten classifiers, computed
from the predictions and probabilities already on disk. **No classifier was
from config import DATA_ROOT  # noqa: E402
retrained, refitted, tuned or modified**, and no prediction was regenerated.
All metrics use the same 384,315 labelled Pill2 voxels as every previous
analysis, with the same classifier names and class labels.

## Geometry

| | |
|---|---|
| Grid (z, y, x) | 128 x 330 x 128 |
| Voxel spacing | **0.32 x 0.32 x 0.32 mm, isotropic** |
| Voxel volume | 0.032768 mm3 |
| Source | Pill2 DICOM: `PixelSpacing` 0.32/0.32, `SliceThickness` 0.32, measured `ImagePositionPatient` step 0.3200 mm |

HD95 and ASSD are symmetric: surface voxels are extracted with a 6-connected
erosion, distances come from a Euclidean transform at the spacing above, and
both directions are pooled before taking the 95th percentile (HD95) or the mean
(ASSD).

## Headline comparison

| Classifier | MCC multiclass | MCC OvR macro | Brier multiclass | HD95 macro (mm) | ASSD macro (mm) | mean abs. rel. volume error |
|---|---:|---:|---:|---:|---:|---:|
| gmm | **0.247** | **0.380** | 1.111 | **17.16** | **5.07** | **1.81** |
| random_forest | 0.247 | 0.257 | **0.809** | 34.04 | 10.91 | 2.67 |
| gradient_boosting | 0.246 | 0.263 | 0.812 | 23.58 | 10.38 | 2.71 |
| knn | 0.245 | 0.255 | 0.823 | 34.11 | 11.16 | 2.65 |
| logistic_regression | 0.217 | 0.292 | 0.871 | 22.10 | 9.93 | 2.67 |
| rbf_svm | 0.213 | 0.291 | 1.015 | 26.36 | 10.00 | 3.01 |
| qda | 0.180 | 0.216 | 1.016 | 28.35 | 10.61 | 3.25 |
| lda | 0.174 | 0.247 | 0.849 | 18.24 | 7.28 | 3.95 |
| ordinal | 0.145 | 0.257 | 1.224 | 22.41 | 10.99 | 4.09 |
| hierarchical | 0.142 | 0.244 | 1.215 | 23.91 | 10.59 | 4.29 |

## Main differences

**The GMM wins every geometric metric and loses the calibration one.** It has
the best MCC one-vs-rest macro (0.380 against 0.292 for the next best), the best
HD95 (17.2 mm), the best ASSD (5.07 mm, roughly half the field) and the smallest
volume error — yet the *worst* multiclass Brier score of the supervised models
(1.111 vs 0.809 for random forest). Its predictions land in the right places but
it is overconfident where it is wrong, which Brier punishes and the hard-label
metrics do not. Its tissue Brier alone is 0.552, the second highest.

**Multiclass MCC is low for everything: 0.14 to 0.25.** Even the best model
agrees with the truth only weakly once chance agreement is removed. The three
tree-based models cluster tightly at 0.245-0.247 with the GMM, so on this metric
the unsupervised model has no advantage; its edge appears only in the per-class
(OvR) form, where class `0` at MCC 0.984 pulls it clear.

**Volume errors are severe and systematic in a consistent direction:**

| Class | relative volume error across the ten models |
|---|---|
| `0.1` | **+736% to +1711%** — massively over-predicted |
| `0.25` | -78% to -93% — almost entirely missed |
| `tissue` | -57% to -99.99% — heavily under-predicted |
| `0` | +3% (gmm) to +457% |
| `0.3` | -43% to +9% — the only class any model gets roughly right |

`0.1` and `tissue` are two views of the same error: tissue voxels are being
labelled `0.1`, inflating one and depleting the other. The GMM is the only model
that keeps class `0` near-exact (+3.0%); every other model over-predicts it by
160-457%.

**Distance metrics separate tissue from the concentrations.** ASSD for `tissue`
is 0.43-1.13 mm for most models (1-3 voxels) but 9.2 mm for the two staged
models, which predict essentially no tissue at all. `0.1` is the worst class
everywhere (ASSD 17-19 mm) because its predicted mask is scattered across the
whole phantom rather than localised.

**The staged models fail distinctly.** `hierarchical` and `ordinal` post a
*negative* tissue MCC (-0.013), meaning their tissue prediction is very slightly
worse than chance, and a -99.992% tissue volume error: out of 347,114 true tissue
voxels they predict roughly 28. Their gate does not fire, as every earlier
analysis showed.

## Limitations

1. **`0.05` and `0.2` have no Pill2 ground truth.** Volume error (relative),
   HD95 and ASSD are undefined and are left blank with a `status` string, never
   filled with a placeholder. Their one-vs-rest MCC is exactly 0 by definition,
   because one side of the comparison is constant — this is reported explicitly
   rather than as a computed value. Note that this contributes two forced zeros
   to every classifier's `mcc_ovr_macro`; the comparison between classifiers is
   still fair because the penalty is identical for all, but the absolute value is
   depressed.
2. **Brier for those two classes measures false confidence only.** With no
   positives, the score simply penalises any probability assigned. It is not
   comparable to the Brier of a class that actually exists, so `brier_macro`
   should be read as a relative ranking, not an absolute calibration figure.
3. **Macro Brier is dominated by `tissue`**, which is 90% of the voxels.
4. **HD95 and ASSD are computed on voxel sets, not anatomical surfaces.** The
   masks are restricted to the 384,315 labelled voxels and, for the poorly
   performing classes, are scattered rather than contiguous. Surface distances
   for such masks are inflated and should be compared between classifiers, not
   interpreted as a boundary error against a real anatomical structure.
5. **MCC, Brier and the distance metrics all rest on the same predictions**, so
   they are not independent evidence; they characterise different aspects of one
   set of outputs.

## Files

| Path | Content |
|---|---|
| `Metrics/Volume_error/volume_error.csv` + `.png` | per class: true/predicted voxels and mm3, signed and absolute error, relative error |
| `Metrics/Brier/brier.csv` + `.png` | per class Brier, mean probability, prevalence |
| `Metrics/MCC/mcc.csv` + `.png` + `mcc_multiclass.png` | one-vs-rest MCC per class, with status |
| `Metrics/HD95/hd95.csv` + `.png` | HD95 in mm, spacing recorded per row |
| `Metrics/ASSD/assd.csv` + `.png` | ASSD in mm, spacing recorded per row |
| `Metrics/summary.csv` | one row per classifier, all metrics combined |
| `Metrics/metric_notes.json` | geometry, spacing provenance, metric definitions |

Existing `Metrics/` results (AUC, IoU, Precision, Recall, F1_Dice, Macro_F1,
Balanced_accuracy, Confusion_matrix, Coverage, Probability_distributions,
probability_maps) were not modified; the script refuses to run if any of the new
folders already exists.
"""

(M / "summary.md").write_text(text)
print(f"wrote {M / 'summary.md'} ({len(text)} chars)")
