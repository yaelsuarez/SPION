#!/usr/bin/env python3
"""Axial companion to the sagittal all-class overviews, for every classifier.

Reads the label volumes already written under Results/Classifiers_probabilities_volumes
and adds one new figure per classifier. Nothing existing is overwritten: the new
files carry an explicit `_axial` suffix, and the GMM's original overview is left
exactly as it is.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from data_treatment.all_models_probability_volumes import (ECHO_MS, LABEL_MAP,  # noqa: E402
                                                           MRI, OUT, PALETTE)

STEM = "all_classes_overview_axial"
N_SLICES = 8


def axial_overview(labels, background, lo, hi, path, title):
    from matplotlib.patches import Patch
    from echoviewer.diagnostics import _figure

    present = np.flatnonzero((labels > 0).sum(axis=(1, 2)) > 0)
    step = max(1, (int(present.max()) - int(present.min())) // (N_SLICES - 1))
    slices = list(range(int(present.min()), int(present.max()) + 1, step))[:N_SLICES]

    rgb = {i: tuple(int(PALETTE[i][j:j + 2], 16) / 255 for j in (1, 3, 5)) for i in LABEL_MAP}
    fig = _figure(figsize=(1.9 * len(slices) + 3.4, 6.4))
    axes = fig.subplots(1, len(slices), squeeze=False)[0]
    for ax, k in zip(axes, slices):
        ax.imshow(background[k], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        plane = labels[k]
        layer = np.zeros(plane.shape + (4,), np.float32)
        for value, colour in rgb.items():
            m = plane == value
            if m.any():
                layer[m, 0], layer[m, 1], layer[m, 2], layer[m, 3] = (*colour, 0.5)
        ax.imshow(layer, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"axial z={k}", fontsize=9)
    fig.legend(handles=[Patch(facecolor=PALETTE[i], alpha=.5, edgecolor="black",
                              label=f"{i} = {c}") for i, c in LABEL_MAP.items()],
               loc="center right", fontsize=9, title="class label")
    fig.suptitle(title, fontsize=13)
    fig.subplots_adjust(right=0.86)
    for suffix in (".svg", ".png"):
        fig.savefig(path.with_suffix(suffix), dpi=130, bbox_inches="tight", facecolor="white")
    fig.clf()


def main() -> int:
    from echoviewer.dataset import get_available_echotimes, load_volume_from_folder

    started = time.time()
    folders = sorted(p for p in OUT.iterdir()
                     if p.is_dir() and (p / "all_clases" / "labels_thresholded.npy").is_file())
    for folder in folders:
        for suffix in (".svg", ".png"):
            target = folder / "all_clases" / f"{STEM}{suffix}"
            assert not target.exists(), f"{target} already exists; refusing to overwrite"

    echo = min(get_available_echotimes(MRI / "Volumes" / "Pill2"),
               key=lambda e: abs(e.value - ECHO_MS))
    background = load_volume_from_folder(echo.path).data.astype(np.float32)
    lo, hi = np.percentile(background, [1, 99.5])
    print(f"background: Pill2 TE {echo.value:g} ms, window {lo:.0f}-{hi:.0f}\n")

    for folder in folders:
        labels = np.load(folder / "all_clases" / "labels_thresholded.npy")
        axial_overview(labels, background, lo, hi,
                       folder / "all_clases" / STEM,
                       f"{folder.name} — all-class label map at alpha 0.5 over the "
                       f"{echo.value:g} ms echo, axial")
        print(f"  {folder.name:20s} {int((labels > 0).sum()):>7,} labelled voxels -> "
              f"{STEM}.svg / .png")
        del labels
    print(f"\n{len(folders)} classifiers, saved under {OUT}\n"
          f"elapsed {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
