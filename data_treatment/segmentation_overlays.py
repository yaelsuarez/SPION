#!/usr/bin/env python3
"""Middle-slice overlays of every segmentation on the first-echo volume.

One SVG per sample, three orthogonal views side by side, every segmentation of
that sample drawn together at alpha 0.5 with a shared class colour scheme.

Plane extraction follows dicom_viewer's own convention, taken from its Volume
class rather than re-derived: axial planes are used as stored, coronal and
sagittal are flipped vertically for display.

Read-only. Original volumes and segmentations are never opened for writing.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT, bootstrap  # noqa: E402
bootstrap()

import numpy as np  # noqa: E402

MRI = DATA_ROOT
SEG = MRI / "Segmentations"
VOLUMES = MRI / "Volumes"
OUT = SEG / "overlays"
SAMPLES = ["Syringes", "3D", "Pill1", "Pill2"]
ALPHA = 0.5
#: Shared with every earlier figure in the project, so a class keeps its colour.
EXTRA_COLORS = {"0.5": "#6ba3d6"}


def plane(volume, view, index):
    """Match dicom_viewer: axial as stored, coronal and sagittal flipped."""
    if view == "Axial":
        return volume[index]
    if view == "Coronal":
        return np.flipud(volume[:, index, :])
    return np.flipud(volume[:, :, index])


def main() -> int:
    from matplotlib.patches import Patch
    from echoviewer.dataset import get_available_echotimes, load_volume_from_folder
    from echoviewer.diagnostics import LABEL_COLORS, _figure

    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    colors = {**LABEL_COLORS, **EXTRA_COLORS}
    report = []

    for sample in SAMPLES:
        # labels come from volumes/, which is where the masks live and which
        # agrees with the fitted stores
        seg_dirs = sorted(p for p in (SEG / sample / "volumes").iterdir() if p.is_dir())
        masks = {}
        for d in seg_dirs:
            label = d.name.split("_", 2)[2]
            masks[label] = np.load(d / "mask.npz")["mask"]

        echoes = get_available_echotimes(VOLUMES / sample)
        first = min(echoes, key=lambda e: e.value)
        background = load_volume_from_folder(first.path).data.astype(np.float32)
        shape = background.shape
        for label, m in masks.items():
            assert m.shape == shape, f"{sample}/{label} mask {m.shape} != volume {shape}"

        lo, hi = np.percentile(background, [1, 99.5])
        middles = {"Axial": shape[0] // 2, "Coronal": shape[1] // 2,
                   "Sagittal": shape[2] // 2}

        fig = _figure(figsize=(16, 6.4))
        axes = fig.subplots(1, 3)
        present_overall = set()
        for ax, view in zip(axes, ("Axial", "Coronal", "Sagittal")):
            index = middles[view]
            base = plane(background, view, index)
            ax.imshow(base, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
            drawn = []
            for label in sorted(masks, key=lambda s: (s == "tissue", s)):
                sl = plane(masks[label], view, index)
                if not sl.any():
                    continue
                drawn.append(label)
                present_overall.add(label)
                colour = colors.get(label, "#ffffff")
                rgb = tuple(int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
                layer = np.zeros(sl.shape + (4,), np.float32)
                layer[..., 0], layer[..., 1], layer[..., 2] = rgb
                layer[..., 3] = np.where(sl, ALPHA, 0.0)
                ax.imshow(layer, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{view} — middle slice {index}", fontsize=11)
            report.append({"sample": sample, "view": view, "slice": index,
                           "classes_visible": ", ".join(drawn) or "none"})

        handles = [Patch(facecolor=colors.get(l, "#ffffff"), alpha=ALPHA, edgecolor="black",
                         label=f"{l}" + (" mg/mL" if l not in ("tissue", "water") else ""))
                   for l in sorted(masks, key=lambda s: (s in ("tissue", "water"), s))]
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=10,
                   frameon=False, title="Segmentation class")
        fig.suptitle(f"{sample} — all segmentations on the first echo "
                     f"(TE = {first.value:g} ms), alpha {ALPHA}", fontsize=14)
        fig.subplots_adjust(bottom=0.16)
        path = OUT / f"{sample}_segmentation_overlay.svg"
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        fig.savefig(path.with_suffix(".png"), dpi=130, bbox_inches="tight", facecolor="white")
        fig.clf()
        print(f"  {sample:9s} {len(masks)} segmentations {sorted(masks)} -> {path.name}")
        for row in report[-3:]:
            print(f"      {row['view']:9s} slice {row['slice']:3d}: {row['classes_visible']}")
        del background, masks

    import pandas as pd
    pd.DataFrame(report).to_csv(OUT / "overlay_index.csv", index=False)
    (OUT / "README.json").write_text(json.dumps({
        "content": "middle-slice overlays of every segmentation, one SVG per sample",
        "background": "first echo time of each sample, grayscale, 1-99.5 percentile window",
        "alpha": ALPHA,
        "class_colors": colors,
        "plane_convention": "dicom_viewer: axial as stored, coronal and sagittal flipped",
        "label_source": ("Segmentations/<sample>/volumes/, which holds the masks and agrees "
                         "with i0_rs and Normalized_data_i0_rs. Note that "
                         "Segmentations/Syringes/segmentations/ names the same segmentation "
                         "'Segmentation_2_0.5' while volumes/ and every fitted store call it "
                         "'0.05'; nothing was renamed."),
        "originals_modified": False,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
    print(f"\nsaved to {OUT}\nelapsed {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
