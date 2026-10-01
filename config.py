"""Single place where every path and every frozen constant is defined.

Nothing else in the repository should contain an absolute path. Scripts that
still do are being migrated; import from here instead:

    from config import VOLUMES, SEGMENTATIONS, RESULTS

Where the data lives
--------------------
Resolved in this order, first hit wins:

1. the ``SPION_QUANT_DATA`` environment variable
2. a file named ``data_root.txt`` next to this module, containing one path
3. ``./data`` relative to the repository root

So a user who clones this repository only has to do one of:

    export SPION_QUANT_DATA=/path/to/their/MRI
    echo /path/to/their/MRI > data_root.txt
    ln -s /path/to/their/MRI data

The frozen constants
--------------------
The normalisation factors, the f(rs) polynomial and the quality thresholds
below were derived from the phantoms used in this study. They are correct for
that dataset and meaningless for another. Anyone re-running this pipeline on
their own data must re-derive them; they are collected here rather than left
scattered through the scripts so that is obvious and easy.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# data root
# --------------------------------------------------------------------------
def _resolve_data_root() -> Path:
    env = os.environ.get("SPION_QUANT_DATA")
    if env:
        return Path(env).expanduser()
    pointer = REPO / "data_root.txt"
    if pointer.is_file():
        text = pointer.read_text().strip()
        if text:
            return Path(text).expanduser()
    return REPO / "data"


DATA_ROOT = _resolve_data_root()

#: raw acquisitions: Volumes/<sample>/<echo_time_ms>/*.dcm
VOLUMES = DATA_ROOT / "Volumes"
#: manual/derived segmentations: Segmentations/<sample>/{segmentations,volumes}/...
SEGMENTATIONS = DATA_ROOT / "Segmentations"
#: voxel-wise fit output, per segmentation and per whole volume
I0_RS = DATA_ROOT / "i0_rs"
#: normalised feature stores
NORMALIZED = DATA_ROOT / "Normalized_data_i0_rs"
#: classifier training, models and metrics
CLASSIFIERS = DATA_ROOT / "Classifiers"
TRAINING = DATA_ROOT / "Training"
#: figures, workbooks and derived volumes
RESULTS = DATA_ROOT / "Results"
#: human and mouse intestine analysis
HUMAN = DATA_ROOT / "human"


#: the original standalone DICOM viewer, imported as the ``dicomview`` package.
#: It is a separate repository; point at it with DICOMVIEW_HOME, otherwise the
#: usual sibling checkout locations are tried.
def _resolve_dicomview() -> Path:
    env = os.environ.get("DICOMVIEW_HOME")
    if env:
        return Path(env).expanduser()
    for candidate in (REPO.parent / "dicom_viewer",
                      REPO.parents[2] / "dicom_viewer",
                      REPO / "dicom_viewer"):
        if (candidate / "dicomview").is_dir():
            return candidate
    return REPO.parent / "dicom_viewer"


DICOMVIEW_HOME = _resolve_dicomview()

#: manuscript figures; separate from the data tree
FIGURES = Path(os.environ.get("SPION_QUANT_FIGURES",
                              DATA_ROOT.parent / "Figures" / "MRI" / "volumes")).expanduser()

#: raw human and mouse intestine acquisitions, a separate collection
HUMAN_SOURCE = Path(os.environ.get("SPION_QUANT_HUMAN_SOURCE",
                                   DATA_ROOT / "human_source")).expanduser()


def bootstrap() -> None:
    """Put this repository and the original viewer on ``sys.path``.

    Scripts that import both ``echoviewer`` and ``dicomview`` call this instead
    of hardcoding either location.
    """
    import sys

    for path in (REPO, DICOMVIEW_HOME):
        entry = str(path)
        if entry not in sys.path:
            sys.path.insert(0, entry)


def require(*paths: Path) -> None:
    """Fail early, and say what is missing and how to point at it."""
    missing = [p for p in paths if not p.exists()]
    if missing:
        listed = "\n  ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"data not found:\n  {listed}\n\n"
            f"DATA_ROOT is currently {DATA_ROOT}\n"
            "Set SPION_QUANT_DATA, write data_root.txt, or symlink ./data "
            "(see config.py)."
        )


# --------------------------------------------------------------------------
# acquisition
# --------------------------------------------------------------------------
#: (z, y, x) in mm; isotropic for the phantom acquisitions in this study
VOXEL_SPACING_MM = (0.32, 0.32, 0.32)
#: multi-echo spin echo, 26 echoes
ECHO_TIMES_MS = tuple(float(8 * i) for i in range(1, 27))


# --------------------------------------------------------------------------
# fitting and quality flags
# --------------------------------------------------------------------------
#: Model A: I0 * exp(-rs * TE)      Model B: I0 * exp(-rs * TE) + B
#: the model with the lower AIC is kept; ties go to A
MAX_RATE_PER_MS = 1.0
MIN_POINTS_NO_BASELINE = 3
MIN_POINTS_WITH_BASELINE = 4
#: rs above this is not identifiable at these echo times
FLAG_HIGH_RS = 0.15
#: three free parameters need at least four informative echoes
MIN_INFORMATIVE = 4


# --------------------------------------------------------------------------
# STUDY-SPECIFIC. Re-derive these for any other dataset.
# --------------------------------------------------------------------------
CLASSES = ("0", "0.05", "0.1", "0.2", "0.25", "0.3", "tissue")

#: label-light I0 references: median I0_selected of each sample's own 0-SPION
#: segmentation (quality_ok voxels). Pill1 has none, so its value is estimated
#: from its 0.2 segmentation using the concentration effect learned from
#: Syringes and 3D only.
NORMALIZATION_REFERENCES = {
    "Syringes": 7301.30,
    "3D": 1069.995,
    "Pill1": 1294.1916772780196,
    "Pill2": 2042.62,
}

#: I0_residual = I0_normalized - f(rs); degree-2 polynomial, highest order first.
#: Fitted on Syringes + 3D + Pill1 only (Pill2 excluded), flag_high_rs == 0,
#: 1,168,086 voxels, R^2 = 0.111.
F_RS_COEFFICIENTS = (-134.01472981222213, 15.60533071525926, 0.9382324838246092)

#: Pill2 ground truth correction: the segmentation originally labelled 0.25 was
#: later established to be 0.2. Applied to labels only, never to predictions.
PILL2_LABEL_CORRECTION = {"0.25": "0.2"}

#: fixed everywhere a random draw is made, so runs are reproducible
SEED = 42


__all__ = [
    "REPO", "DATA_ROOT", "VOLUMES", "SEGMENTATIONS", "I0_RS", "NORMALIZED",
    "CLASSIFIERS", "TRAINING", "RESULTS", "HUMAN", "require",
    "VOXEL_SPACING_MM", "ECHO_TIMES_MS", "MAX_RATE_PER_MS",
    "MIN_POINTS_NO_BASELINE", "MIN_POINTS_WITH_BASELINE", "FLAG_HIGH_RS",
    "MIN_INFORMATIVE", "CLASSES", "NORMALIZATION_REFERENCES",
    "F_RS_COEFFICIENTS", "PILL2_LABEL_CORRECTION", "SEED",
]


if __name__ == "__main__":
    print(f"DATA_ROOT = {DATA_ROOT}   (exists: {DATA_ROOT.exists()})")
    for name in ("VOLUMES", "SEGMENTATIONS", "I0_RS", "NORMALIZED",
                 "CLASSIFIERS", "TRAINING", "RESULTS", "HUMAN"):
        path = globals()[name]
        print(f"  {name:15s} {'ok ' if path.exists() else '-- '} {path}")
