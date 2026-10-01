#!/usr/bin/env python3
"""Standalone colourbar swatches at exact physical size, as SVG.

Only the ramp: no ticks, labels, frame or padding, transparent background. Saved
without bbox_inches="tight", which crops to content and would change the size.
"""

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from pathlib import Path
from config import FIGURES  # noqa: E402

FIGURES = FIGURES
MM = 1 / 25.4

#: (filename, colormap, width mm, height mm, orientation)
BARS = [
    ("inferno_colorbar_10x35mm.svg", "inferno", 10, 35, "vertical"),
    ("gray_colorbar_50x30mm.svg", "gray", 50, 30, "horizontal"),
]


def bar(path, cmap, width_mm, height_mm, orientation):
    fig = Figure(figsize=(width_mm * MM, height_mm * MM))
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])          # axes fill the figure exactly
    ramp = (np.linspace(1, 0, 512).reshape(-1, 1) if orientation == "vertical"
            else np.linspace(0, 1, 512).reshape(1, -1))
    ax.imshow(ramp, cmap=cmap, aspect="auto", interpolation="bilinear")
    ax.set_axis_off()
    fig.savefig(path, format="svg", transparent=True)
    return path


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for filename, cmap, w, h, orientation in BARS:
        path = bar(FIGURES / filename, cmap, w, h, orientation)
        print(f"saved {path.name}  ({w} x {h} mm, {cmap}, {orientation})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
