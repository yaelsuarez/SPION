#!/usr/bin/env python3
"""Write all_classes_overview.svg from the saved all-class volume.

Redraws the same overview from the .npy files already in all_clases/ and writes
the SVG alongside the existing PNG. No volume, DICOM or metric is recomputed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from data_treatment.gmm_all_classes_volume import OUT, build_overview  # noqa: E402


def main() -> int:
    labels = np.load(OUT / "labels_thresholded.npy")
    max_prob = np.load(OUT / "max_probability.npy")
    valid = np.load(OUT / "valid_mask.npy")
    print(f"labels {labels.shape}, {int((labels > 0).sum()):,} labelled voxels")
    for path in build_overview(labels, max_prob, valid, OUT / "all_classes_overview.png"):
        print(f"wrote {path.name}  ({path.stat().st_size/1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
