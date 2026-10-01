# SPION_Quant

Voxel-wise quantification of superparamagnetic iron oxide nanoparticle (SPION)
concentration from multi-echo MRI, and the analysis behind the accompanying
manuscript.

The repository contains two things:

| | |
|---|---|
| **`echoviewer/`** | a general-purpose engine — DICOM loading, an echo-time viewer, 3D segmentation, voxel-wise T2 relaxometry, export. Reusable on any multi-echo dataset. |
| **`data_treatment/`, `human/`, `run_*.py`** | the specific analysis for this manuscript. These encode decisions, class labels and calibration constants that belong to *these* phantoms. They reproduce the published results; they are not a general tool. |

If you came here to process your own data, start with `echoviewer/`. 

If you came to check how a figure or number in the paper was produced, look in
`data_treatment/` — `code_inventory.pdf` lists every file with its purpose.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate    # Python 3.11+
pip install -r requirements.txt
```

`PyQt5` is only needed for the interactive viewer; everything else runs headless.

## Point the code at your data

Every path resolves through `config.py`. Choose one:

```bash
export SPION_QUANT_DATA=/path/to/your/MRI      # environment variable
echo /path/to/your/MRI > data_root.txt         # or a pointer file
ln -s /path/to/your/MRI data                   # or a symlink
```

Check what it found:

```bash
python config.py
```

## Expected data layout

```
<DATA_ROOT>/
  Volumes/<sample>/<echo_time_ms>/*.dcm        raw multi-echo acquisitions
  Segmentations/<sample>/
      segmentations/Segmentation_<n>_<material>/    sparse voxel lists
      volumes/Segmentation_<n>_<material>/mask.npz  dense boolean masks
  i0_rs/                                        voxel-wise fit output
  Normalized_data_i0_rs/                        normalised features
  Classifiers/, Training/, Results/, human/     models, metrics, figures
```

Two conventions the code relies on:

- **echo-time folders are named by their TE in ms** (`8.0`, `16.0`, …) and are
  sorted numerically, not lexically.
- **material identity lives in the segmentation folder name**
  (`Segmentation_4_0.25` → material `0.25`). It is always read from the folder,
  never from a CSV column — pandas infers `"0"` as an integer and `"tissue"` as a
  string, which silently produces a mixed-type target.

Volumes are indexed `(z, y, x)` = (slice, row, column). NIfTI masks in the
human/mouse dataset are stored transposed and are read with
`np.transpose(mask, (2, 1, 0))`; this was verified against the source
`results.xlsx`, not assumed.

---

## The method, in short

**Fitting.** Each voxel's decay is fitted twice — `I0·exp(−rs·TE)` and
`I0·exp(−rs·TE) + B` — and the model with the lower AIC is kept, ties going to
the simpler one. That gives `rs_selected`, `I0_selected`, `B_selected` per voxel.

**Normalisation, two steps.**

1. *Scale*: `I0_normalized = I0_selected / reference`, where the reference is the
   median `I0_selected` of that sample's own 0-SPION segmentation.
2. *rs-residual*: `I0_residual = I0_normalized − f(rs)`, removing the systematic
   dependence of amplitude on relaxation rate so the two features are
   complementary rather than partly redundant.

**Classification.** Ten classifiers on the fixed feature set
`rs_selected + I0_residual`, trained on three phantoms with the fourth held out
entirely, validated by `GroupKFold` on segmentation identity so no segmentation
is split across folds.

## Reproducing the analysis

Roughly in order. Each script is standalone and prints what it wrote.

```
run_echo_viewer.py                     inspect and segment interactively
echoviewer/relaxometry.py              voxel-wise fitting  ->  i0_rs/
data_treatment/verify_labellight.py    check the normalisation before applying it
data_treatment/apply_labellight_normalization.py
data_treatment/train_baseline_classifier.py
data_treatment/classifier_benchmark.py            + _extension.py
data_treatment/auc_metrics.py  iou_metrics.py  segmentation_metrics.py
data_treatment/organise_metrics.py     collect everything into Metrics/
```

Tests:

```bash
pytest tests/
```

---

## Constants you must re-derive for other data

Collected in `config.py` with their provenance. They are correct for the
phantoms in this study and meaningless elsewhere:

- `NORMALIZATION_REFERENCES` — per-sample I0 scale factors.
- `F_RS_COEFFICIENTS` — the degree-2 polynomial subtracted in step 2. Fitted on
  Syringes + 3D + Pill1 only, R² = 0.111. It is a shallow detrend, not a strong
  correction.
- `FLAG_HIGH_RS = 0.15` — above this, three parameters are not identifiable at
  these echo times.
- `CLASSES` — the concentration labels of this phantom set.

## Known limitations

Stated plainly, because they bound what the published results mean.

- **Tissue and low SPION overlap intrinsically.** Every classifier assigns
  57–100% of tissue voxels to the two lowest concentrations. This is a property
  of the measurement at this noise level, not of the algorithm, and no model or
  feature tested resolved it.
- **One phantom provides Pill1's normalisation reference indirectly.** It has no
  0-SPION region, so its factor is transferred using a concentration effect
  learned from the other two phantoms. That makes the scheme "label-light"
  rather than label-free.
- **A ground-truth label was corrected mid-analysis.** The Pill2 segmentation
  originally labelled `0.25` is `0.2`. Results exist in both forms
  (`Metrics/` and `Metrics_corrected_labels/`); the corrected version supersedes
  the other, and the correction applies to labels only, never to predictions.
- **Some classes come from a single segmentation**, so grouped cross-validation
  cannot validate them.
- **The analysis scripts have no test coverage.** `pytest` exercises
  `echoviewer` thoroughly, but `data_treatment/` and `human/` are validated only
  by the outputs they produced for the manuscript.

## Repository map

See `code_inventory.pdf` for all 86 files with purposes and paths.
`docs/ECHOVIEWER.md` is the detailed developer documentation for the viewer and
segmentation tooling — controls, the 2D-to-3D segmentation design, export
formats, and the UI behaviours that were deliberate rather than accidental.

## Citation and licence

Released under the **MIT License** — see `LICENSE`. You may use, modify and
redistribute this code, including commercially, provided the copyright notice
and licence text are retained. It comes with no warranty.

If you use this code in published work, please cite the accompanying
manuscript.
