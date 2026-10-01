"""Coronal montage of the Pill2 full-volume prediction, every 10 slices."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT  # noqa: E402
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

MRI = DATA_ROOT
VOL = MRI / "i0_rs" / "Volume_for_testing" / "Pill2_normalized"
OUT = MRI / "Training"
CLASSES = ["0", "0.05", "0.1", "0.2", "0.25", "0.3", "tissue"]

def main():
    import echoviewer  # noqa: F401
    from echoviewer.diagnostics import LABEL_COLORS, _figure, _save
    from echoviewer import EchoSeries

    pred = np.load(OUT / "pill2_fullvolume_prediction.npy")
    valid = np.load(VOL / "fit_success.npy") > 0
    s = EchoSeries.from_sample(MRI / "Volumes", "Pill2", cache_size=1)
    raw = s.volume(0).data.copy(); s.close()

    ys = np.flatnonzero(valid.sum(axis=(0, 2)) > 0)
    slices = list(range(int(ys.min()), int(ys.max()) + 1, 10))
    print(f"coronal slices with signal: y {ys.min()}-{ys.max()}  -> {len(slices)} slices every 10")

    colors = [LABEL_COLORS[c] for c in CLASSES]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(0.5, len(CLASSES) + 1.5), cmap.N)

    cols = 6
    rows = int(np.ceil(len(slices) / cols))
    fig = _figure(figsize=(3.1 * cols, 3.5 * rows))
    axes = fig.subplots(rows, cols, squeeze=False).ravel()
    for ax, j in zip(axes, slices):
        base = np.flipud(raw[:, j, :])
        p = np.flipud(pred[:, j, :])
        vmax = float(np.percentile(base[base > 0], 99)) if (base > 0).any() else 1
        ax.imshow(base, cmap="gray", vmax=vmax, interpolation="nearest")
        ax.imshow(np.ma.masked_where(p == 0, p), cmap=cmap, norm=norm,
                  alpha=0.75, interpolation="nearest")
        ax.set_title(f"coronal y = {j}   ({int((p>0).sum()):,} px)", fontsize=9)
        ax.set_axis_off()
    for ax in axes[len(slices):]:
        ax.set_axis_off()
    fig.legend(handles=[Patch(facecolor=LABEL_COLORS[c], label=c) for c in CLASSES],
               loc="lower center", ncol=7, fontsize=11, frameon=False,
               bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Pill2 full-volume prediction — coronal slices every 10, on the first-echo image",
                 fontsize=15)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    _save(fig, OUT / "pill2_fullvolume_coronal_every10.png", [])
    print("saved", OUT / "pill2_fullvolume_coronal_every10.png")

main()
