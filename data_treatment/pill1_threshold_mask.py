#!/usr/bin/env python3
"""Pill1 I0/rs threshold mask: IoU against segmentation 0.2, and two figures.

The mask is 1025 < I0 < 2000 and 0.05 < rs < 0.100 on the AIC-selected maps,
restricted to fit_success. Read-only throughout; nothing is refitted.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT, FIGURES  # noqa: E402

import numpy as np  # noqa: E402

MRI = DATA_ROOT
FIGURES = FIGURES
FITS = MRI / "i0_rs" / "Volumes" / "Pill1"
SEG = MRI / "Segmentations" / "Pill1" / "volumes" / "Segmentation_1_0.2" / "mask.npz"
I0_RANGE = (1025, 2000)
RS_RANGE = (0.05, 0.100)
ECHO_MS = 8.0
SEG_RGB = (0.85, 0.15, 0.15)
MASK_RGB = (0.15, 0.35, 0.95)


def load():
    valid = np.load(FITS / "fit_success_mask.npy") > 0
    choice = np.load(FITS / "model_choice_map.npy")
    I0 = np.where(choice == 1, np.load(FITS / "I0_map_noB.npy"),
                  np.load(FITS / "I0_map_B.npy"))
    rs = np.where(choice == 1, np.load(FITS / "rs_map_noB.npy"),
                  np.load(FITS / "rs_map_B.npy"))
    mask = (valid & np.isfinite(I0) & np.isfinite(rs)
            & (I0 > I0_RANGE[0]) & (I0 < I0_RANGE[1])
            & (rs > RS_RANGE[0]) & (rs < RS_RANGE[1]))
    seg = np.load(SEG)["mask"]
    return mask, seg


def overlay(ax, sl, rgb, alpha=0.5):
    layer = np.zeros(sl.shape + (4,), np.float32)
    layer[..., 0], layer[..., 1], layer[..., 2] = rgb
    layer[..., 3] = np.where(sl, alpha, 0.0)
    ax.imshow(layer, interpolation="nearest")


def main() -> int:
    from matplotlib.patches import Patch
    from echoviewer.dataset import get_available_echotimes, load_volume_from_folder
    from echoviewer.diagnostics import _figure
    from data_treatment.rs_map_overview import VIEWS, plane

    FIGURES.mkdir(parents=True, exist_ok=True)
    mask, seg = load()
    inter, union = int((mask & seg).sum()), int((mask | seg).sum())
    iou = inter / union
    print(f"segmentation 0.2 : {int(seg.sum()):,}")
    print(f"threshold mask   : {int(mask.sum()):,}")
    print(f"intersection     : {inter:,}")
    print(f"union            : {union:,}")
    print(f"IoU              : {iou:.4f}")
    print(f"recall / precision: {inter/seg.sum():.4f} / {inter/mask.sum():.4f}")

    echo = min(get_available_echotimes(MRI / "Volumes" / "Pill1"),
               key=lambda e: abs(e.value - ECHO_MS))
    background = load_volume_from_folder(echo.path).data.astype(np.float32)
    lo, hi = np.percentile(background, [1, 99.5])

    # ---- mask vs segmentation, three orthogonal views ---------------------
    both = mask | seg
    picks = {"Axial": int(both.sum(axis=(1, 2)).argmax()),
             "Coronal": int(both.sum(axis=(0, 2)).argmax()),
             "Sagittal": int(both.sum(axis=(0, 1)).argmax())}
    fig = _figure(figsize=(16, 6.6))
    axes = fig.subplots(1, 3)
    for ax, view in zip(axes, VIEWS):
        index = picks[view]
        ax.imshow(plane(background, view, index), cmap="gray", vmin=lo, vmax=hi,
                  interpolation="nearest")
        overlay(ax, plane(seg, view, index), SEG_RGB)
        overlay(ax, plane(mask, view, index), MASK_RGB)
        ms, ss = plane(mask, view, index), plane(seg, view, index)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{view} — slice {index}\nseg {int(ss.sum()):,}  "
                     f"mask {int(ms.sum()):,}  both {int((ms & ss).sum()):,}", fontsize=10)
    fig.legend(handles=[Patch(facecolor=SEG_RGB, alpha=.5, label="segmentation 0.2"),
                        Patch(facecolor=MASK_RGB, alpha=.5, label="threshold mask"),
                        Patch(facecolor=(0.5, 0.25, 0.55), alpha=.85, label="overlap (both)")],
               loc="lower center", ncol=3, fontsize=11, frameon=False)
    fig.suptitle(f"Pill1 — threshold mask ({I0_RANGE[0]}<I0<{I0_RANGE[1]}, "
                 f"{RS_RANGE[0]}<rs<{RS_RANGE[1]}) vs segmentation 0.2   |   "
                 f"IoU = {iou:.4f}", fontsize=13)
    fig.subplots_adjust(bottom=0.14)
    fig.savefig(FIGURES / "Pill1_mask_vs_segmentation.png", dpi=140,
                bbox_inches="tight", facecolor="white")
    fig.clf()

    # ---- mask only, peak axial slice and +/- 5 ----------------------------
    per_slice = mask.sum(axis=(1, 2))
    best = int(per_slice.argmax())
    fig = _figure(figsize=(12, 6.2))
    axes = fig.subplots(1, 3)
    for ax, index in zip(axes, [best - 5, best, best + 5]):
        ax.imshow(background[index], cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        overlay(ax, mask[index], MASK_RGB)
        ax.set_xticks([]); ax.set_yticks([])
        tag = " (peak)" if index == best else f" ({index - best:+d})"
        ax.set_title(f"Axial slice {index}{tag} — {int(mask[index].sum()):,} mask voxels",
                     fontsize=10)
    fig.legend(handles=[Patch(facecolor=MASK_RGB, alpha=.5,
                              label=f"threshold mask: {I0_RANGE[0]} < I0 < {I0_RANGE[1]} "
                                    f"and {RS_RANGE[0]} < rs < {RS_RANGE[1]}")],
               loc="lower center", fontsize=11, frameon=False)
    fig.suptitle("Pill1 — threshold mask on the 8 ms echo, peak axial slice and ±5",
                 fontsize=13)
    fig.subplots_adjust(bottom=0.12)
    for suffix in (".svg", ".png"):
        fig.savefig(FIGURES / f"Pill1_threshold_mask_axial{suffix}", dpi=140,
                    bbox_inches="tight", facecolor="white")
    fig.clf()
    print(f"\nfigures saved to {FIGURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
