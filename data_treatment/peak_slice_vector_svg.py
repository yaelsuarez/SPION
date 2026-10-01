#!/usr/bin/env python3
"""Fully vector version of the peak-slice montage: voxels as SVG rects.

The matplotlib montage embeds each panel as a raster image. Re-drawing those
panels through matplotlib as vector geometry is not viable: pcolormesh writes a
complete path per voxel with no reuse, which extrapolates to roughly 590 MB for
this figure. So the SVG is emitted directly here.

Two lossless compactions keep the file usable:

* Probabilities are quantised to 256 levels. A matplotlib colormap has exactly
  256 entries, so this is the same quantisation the raster already applies -
  nothing visible is lost.
* Voxels of equal colour adjacent in a row are merged into one rect, and rects
  are grouped under a `<g fill=...>` so the colour is written once per group.

Every voxel is still an editable vector shape. Read-only: reads the saved
full-volume predictions and writes one SVG.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = DATA_ROOT / "Classifiers"
PROB_DIR = OUT / "probability_maps"
TARGET = PROB_DIR / "peak_slice_comparison_vector.svg"
ORDER = ["logistic_regression", "random_forest", "gradient_boosting", "rbf_svm", "knn",
         "lda", "qda", "gmm", "hierarchical", "ordinal"]
SHAPE = (128, 330, 128)  # z, y, x

PANEL_W = 150.0          # panel width in px; heights follow the true aspect
GAP_X, GAP_Y = 10.0, 26.0
LEFT, TOP, RIGHT, BOTTOM = 104.0, 74.0, 76.0, 20.0
BLOCK_GAP = 54.0
LEVELS = 256


def palette():
    from matplotlib import colormaps
    rgba = colormaps["magma"](np.arange(LEVELS) / (LEVELS - 1))
    return ["#%02x%02x%02x" % tuple(int(round(v * 255)) for v in c[:3]) for c in rgba]


def panel_rects(img, colors, scale_x, scale_y):
    """Row-merged rects for one plane. img may contain NaN outside the volume."""
    q = np.where(np.isfinite(img), np.rint(np.nan_to_num(img) * (LEVELS - 1)), -1).astype(np.int16)
    by_colour: dict[int, list[str]] = {}
    for r, row in enumerate(q):
        change = np.r_[True, row[1:] != row[:-1]]
        starts = np.flatnonzero(change)
        ends = np.r_[starts[1:], len(row)]
        for s, e in zip(starts, ends):
            level = int(row[s])
            if level < 0:
                continue
            by_colour.setdefault(level, []).append(
                f'<rect x="{s * scale_x:.3f}" y="{r * scale_y:.3f}"'
                f' width="{(e - s) * scale_x:.3f}" height="{scale_y:.3f}"/>')
    return "".join(f'<g fill="{colors[k]}">{"".join(v)}</g>' for k, v in sorted(by_colour.items()))


def esc(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main() -> int:
    colors = palette()
    classes = json.loads((OUT / ORDER[0] / "metrics.json").read_text())["classes"]
    names = [n for n in ORDER if (OUT / n / "pill2_fullvolume_predictions.csv.gz").is_file()]

    # peak slice per class, from the mean over all methods (same rule as the PNG)
    prof_y = {c: np.zeros(SHAPE[1]) for c in classes}
    prof_x = {c: np.zeros(SHAPE[2]) for c in classes}
    for name in names:
        d = pd.read_csv(OUT / name / "pill2_fullvolume_predictions.csv.gz")
        for c in classes:
            gy = d.groupby("y")[f"p_{c}"].sum(); gx = d.groupby("x")[f"p_{c}"].sum()
            prof_y[c][gy.index.to_numpy()] += gy.to_numpy()
            prof_x[c][gx.index.to_numpy()] += gx.to_numpy()
        del d
    coronal = {c: int(np.argmax(prof_y[c])) for c in classes}
    sagittal = {c: int(np.argmax(prof_x[c])) for c in classes}
    for c in classes:
        print(f"  {c:7s} coronal y={coronal[c]:3d}   sagittal x={sagittal[c]:3d}")

    cor_h = PANEL_W * SHAPE[0] / SHAPE[2]           # 128 rows over 128 cols
    sag_h = PANEL_W * SHAPE[0] / SHAPE[1]           # 128 rows over 330 cols
    width = LEFT + len(classes) * PANEL_W + (len(classes) - 1) * GAP_X + RIGHT
    cor_block = len(names) * (cor_h + GAP_Y)
    sag_block = len(names) * (sag_h + GAP_Y)
    height = TOP + cor_block + BLOCK_GAP + 34 + sag_block + BOTTOM

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'shape-rendering="crispEdges">',
        f'<rect width="{width:.0f}" height="{height:.0f}" fill="white"/>',
        '<g font-family="DejaVu Sans, Helvetica, Arial, sans-serif" text-anchor="middle">',
        f'<text x="{width/2:.0f}" y="26" font-size="20">Pill2 complete volume — class '
        f'probability at each class’s peak slice, {len(names)} methods</text>',
        f'<text x="{width/2:.0f}" y="48" font-size="12" fill="#444">every voxel is a vector '
        f'rect; identical slices and identical 0–1 scale in every panel</text>',
        '</g>',
    ]

    def column_x(i):
        return LEFT + i * (PANEL_W + GAP_X)

    for view, picks, ph in (("coronal", coronal, cor_h), ("sagittal", sagittal, sag_h)):
        y0 = TOP if view == "coronal" else TOP + cor_block + BLOCK_GAP + 34
        label_y = y0 - 30 if view == "coronal" else y0 - 30
        parts.append('<g font-family="DejaVu Sans, Helvetica, Arial, sans-serif">')
        parts.append(f'<text x="{LEFT:.0f}" y="{label_y - 18:.0f}" font-size="15" '
                     f'font-weight="bold">{view} — peak-probability slice per class, '
                     f'identical across methods</text>')
        for i, c in enumerate(classes):
            key = "y" if view == "coronal" else "x"
            parts.append(f'<text x="{column_x(i) + PANEL_W/2:.1f}" y="{label_y:.0f}" '
                         f'font-size="13" text-anchor="middle">{esc(c)} '
                         f'({key}={picks[c]})</text>')
        parts.append("</g>")

    for r, name in enumerate(names):
        d = pd.read_csv(OUT / name / "pill2_fullvolume_predictions.csv.gz")
        z = d.z.to_numpy(np.intp); y = d.y.to_numpy(np.intp); x = d.x.to_numpy(np.intp)
        for view, picks, ph in (("coronal", coronal, cor_h), ("sagittal", sagittal, sag_h)):
            y0 = TOP if view == "coronal" else TOP + cor_block + BLOCK_GAP + 34
            top = y0 + r * (ph + GAP_Y)
            parts.append('<g font-family="DejaVu Sans, Helvetica, Arial, sans-serif">'
                         f'<text x="{LEFT - 10:.0f}" y="{top + ph/2 + 4:.1f}" font-size="11" '
                         f'text-anchor="end">{esc(name)}</text></g>')
            for i, c in enumerate(classes):
                p = d[f"p_{c}"].to_numpy(np.float32)
                if view == "coronal":
                    m = y == picks[c]
                    img = np.full((SHAPE[0], SHAPE[2]), np.nan, np.float32)
                    img[z[m], x[m]] = p[m]
                else:
                    m = x == picks[c]
                    img = np.full((SHAPE[0], SHAPE[1]), np.nan, np.float32)
                    img[z[m], y[m]] = p[m]
                img = np.flipud(img)
                sx = PANEL_W / img.shape[1]
                sy = ph / img.shape[0]
                parts.append(f'<g transform="translate({column_x(i):.1f},{top:.1f})">')
                parts.append(f'<rect width="{PANEL_W:.1f}" height="{ph:.1f}" fill="white" '
                             f'stroke="#333" stroke-width="0.8"/>')
                parts.append(panel_rects(img, colors, sx, sy))
                parts.append(f'<rect width="{PANEL_W:.1f}" height="{ph:.1f}" fill="none" '
                             f'stroke="#333" stroke-width="0.8"/>')
                parts.append("</g>")
        del d
        print(f"  drawn {name}")

    bar_x = width - RIGHT + 16
    bar_y, bar_h = TOP, cor_block - GAP_Y
    stops = "".join(f'<stop offset="{i/16:.4f}" stop-color="{colors[int(i/16*(LEVELS-1))]}"/>'
                    for i in range(17))
    parts.append(f'<defs><linearGradient id="scale" x1="0" y1="1" x2="0" y2="0">{stops}'
                 f'</linearGradient></defs>')
    parts.append(f'<rect x="{bar_x:.0f}" y="{bar_y:.0f}" width="16" height="{bar_h:.0f}" '
                 f'fill="url(#scale)" stroke="#333" stroke-width="0.8"/>')
    parts.append('<g font-family="DejaVu Sans, Helvetica, Arial, sans-serif" font-size="11">')
    for frac, lbl in ((0.0, "0.0"), (0.25, "0.25"), (0.5, "0.5"), (0.75, "0.75"), (1.0, "1.0")):
        ty = bar_y + bar_h - frac * bar_h
        parts.append(f'<text x="{bar_x + 21:.0f}" y="{ty + 4:.1f}">{lbl}</text>')
    parts.append(f'<text transform="translate({bar_x - 8:.0f},{bar_y + bar_h/2:.0f}) '
                 f'rotate(-90)" text-anchor="middle">predicted probability</text>')
    parts.append("</g></svg>")

    TARGET.write_text("".join(parts))
    size = TARGET.stat().st_size
    print(f"\nsaved {TARGET}\n{size/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
