#!/usr/bin/env python3
"""Peak-probability slice per class, every method, coronal and sagittal.

For each class the slice carrying the most probability mass is chosen ONCE, from
the mean over all methods, and then reused for every method. Letting each method
pick its own slice would put different anatomy behind every panel and make the
row-to-row comparison meaningless.

Probability volumes are never materialised: with ten methods that would be about
1.5 GB. Peak slices come from 1-D profiles built by groupby, and only the chosen
planes are then rebuilt.

Read-only: reads the saved full-volume predictions and writes a single PNG.
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
ORDER = ["logistic_regression", "random_forest", "gradient_boosting", "rbf_svm", "knn",
         "lda", "qda", "gmm", "hierarchical", "ordinal"]
SHAPE = (128, 330, 128)  # z, y, x
TARGET = PROB_DIR / "peak_slice_comparison.png"
PREVIOUS = PROB_DIR / "peak_slice_comparison_first_five.png"


def main(threshold: float | None = None, dark: bool = False) -> int:
    from echoviewer.diagnostics import _figure, _save

    # A threshold run keeps the unthresholded peak slices, so all the figures
    # show the same anatomy and can be read side by side. At >= 0.5 only one
    # class can clear the bar, so masking p_c is the same as masking on the
    # winning class.
    cmap = "inferno" if dark else "magma"
    face, fg = ("black", "white") if dark else ("white", "black")
    target = TARGET
    if threshold is not None:
        target = PROB_DIR / "cut" / f"{threshold:g}" / "peak_slice_comparison.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"threshold {threshold} -> {target.parent}")
    if dark:
        target = PROB_DIR / "black" / target.name
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"dark background, {cmap} -> {target.parent}")

    if threshold is None and not dark and TARGET.is_file() and not PREVIOUS.is_file():
        PREVIOUS.write_bytes(TARGET.read_bytes())
        print(f"preserved the five-classifier montage as {PREVIOUS.name}")

    classes = json.loads((OUT / ORDER[0] / "metrics.json").read_text())["classes"]
    names = [n for n in ORDER if (OUT / n / "pill2_fullvolume_predictions.csv.gz").is_file()]
    print(f"{len(names)} methods, {len(classes)} classes")

    # pass 1: 1-D probability profiles per class, to locate each class's peak slice
    prof_y = {c: np.zeros(SHAPE[1]) for c in classes}
    prof_x = {c: np.zeros(SHAPE[2]) for c in classes}
    for name in names:
        d = pd.read_csv(OUT / name / "pill2_fullvolume_predictions.csv.gz")
        for c in classes:
            gy = d.groupby("y")[f"p_{c}"].sum()
            gx = d.groupby("x")[f"p_{c}"].sum()
            prof_y[c][gy.index.to_numpy()] += gy.to_numpy()
            prof_x[c][gx.index.to_numpy()] += gx.to_numpy()
        del d
    coronal = {c: int(np.argmax(prof_y[c])) for c in classes}
    sagittal = {c: int(np.argmax(prof_x[c])) for c in classes}
    for c in classes:
        print(f"  {c:7s} coronal y={coronal[c]:3d}   sagittal x={sagittal[c]:3d}")

    # pass 2: rebuild only the chosen planes
    planes = {}
    for name in names:
        d = pd.read_csv(OUT / name / "pill2_fullvolume_predictions.csv.gz")
        z = d.z.to_numpy(np.intp); y = d.y.to_numpy(np.intp); x = d.x.to_numpy(np.intp)
        for c in classes:
            p = d[f"p_{c}"].to_numpy(np.float32)
            if threshold is not None:
                p = np.where(p >= threshold, p, np.nan)
            m = y == coronal[c]
            img = np.full((SHAPE[0], SHAPE[2]), np.nan, np.float32)
            img[z[m], x[m]] = p[m]
            planes[(name, c, "coronal")] = img
            m = x == sagittal[c]
            img = np.full((SHAPE[0], SHAPE[1]), np.nan, np.float32)
            img[z[m], y[m]] = p[m]
            planes[(name, c, "sagittal")] = img
        del d

    fig = _figure(figsize=(3.0 * len(classes), (2.6 + 1.35) * len(names) + 1.6))
    top, bottom = fig.subfigures(2, 1, height_ratios=[2.6 * len(names), 1.35 * len(names)])
    im = None
    for sub, view, picks in ((top, "coronal", coronal), (bottom, "sagittal", sagittal)):
        axes = sub.subplots(len(names), len(classes), squeeze=False)
        for r, name in enumerate(names):
            for col, c in enumerate(classes):
                ax = axes[r][col]
                im = ax.imshow(np.flipud(planes[(name, c, view)]), cmap=cmap,
                               vmin=0, vmax=1, interpolation="nearest")
                # NaN is transparent, so voxels outside the volume take the axes
                # colour: the background reads as a single continuous field.
                ax.set_facecolor(face)
                for spine in ax.spines.values():
                    spine.set_color(fg)
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(f"{c}   ({'y' if view == 'coronal' else 'x'}={picks[c]})",
                                 fontsize=9, color=fg)
                if col == 0:
                    ax.set_ylabel(name, fontsize=7, color=fg)
        sub.set_facecolor(face)
        sub.suptitle(f"{view} — peak-probability slice per class, identical across methods",
                     fontsize=12, y=0.995 if view == "sagittal" else 0.975, color=fg)
    bar = fig.colorbar(im, cax=fig.add_axes([0.925, 0.06, 0.011, 0.88]))
    bar.set_label("predicted probability (identical 0–1 scale in every panel)", fontsize=9,
                  color=fg)
    bar.ax.tick_params(colors=fg)
    bar.outline.set_edgecolor(fg)
    cut = "" if threshold is None else f", showing only probability ≥ {threshold:g}"
    fig.suptitle("Pill2 complete volume — class probability at each class's peak slice, "
                 f"{len(names)} methods{cut}", fontsize=15, color=fg)
    fig.set_facecolor(face)
    svg = target.with_suffix(".svg")
    for path in (svg, target):
        fig.savefig(path, dpi=110, bbox_inches="tight", facecolor=face)
    fig.clf()
    print(f"\nsaved {target}\nsaved {svg}")
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    dark = "dark" in args
    values = [a for a in args if a != "dark"]
    raise SystemExit(main(float(values[0]) if values else None, dark))
