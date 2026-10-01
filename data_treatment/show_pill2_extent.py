"""Show what the 'full volume' prediction actually covers, in all three planes."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT  # noqa: E402
import numpy as np, pandas as pd

MRI = DATA_ROOT
VOL = MRI / "i0_rs" / "Volume_for_testing" / "Pill2_normalized"
SEG = MRI / "Normalized_data_i0_rs" / "Segmentations" / "Pill2"
PRED = MRI / "Training" / "pill2_fullvolume_prediction.npy"
OUT = MRI / "Training"


def main():
    import echoviewer  # noqa: F401
    from echoviewer.diagnostics import _figure, _save
    from echoviewer import EchoSeries

    valid = np.load(VOL / "fit_success.npy") > 0
    pred = np.load(PRED)
    shape = valid.shape

    # union of the five labelled segmentations
    segmask = np.zeros(shape, bool)
    for p in sorted(q for q in SEG.iterdir() if q.is_dir()):
        d = pd.read_csv(p / "voxels.csv.gz", usecols=["x", "y", "z"])
        segmask[d.z, d.y, d.x] = True

    print(f"volume shape                      {shape}  = {np.prod(shape):,} voxels")
    print(f"predicted (valid/Otsu foreground) {int(valid.sum()):,}  ({100*valid.mean():.2f}% of volume)")
    print(f"union of the 5 segmentations      {int(segmask.sum()):,}")
    print(f"predicted AND segmented           {int((valid&segmask).sum()):,}")
    print(f"predicted but NOT segmented       {int((valid&~segmask).sum()):,}  "
          f"<- extra coverage the full volume adds")
    print(f"segmented but NOT predicted       {int((segmask&~valid).sum()):,}  "
          f"<- below the volume's Otsu level")
    print(f"prediction non-zero voxels        {int((pred > 0).sum()):,}")

    for name, m in (("valid/predicted", valid), ("segmentation union", segmask)):
        z, y, x = np.nonzero(m)
        print(f"  {name:20s} extent  z {z.min()}-{z.max()}   y {y.min()}-{y.max()}   x {x.min()}-{x.max()}")

    # first-echo image for context
    s = EchoSeries.from_sample(MRI / "Volumes", "Pill2", cache_size=1)
    raw = s.volume(0).data.copy()
    s.close()

    k = int(np.argmax(valid.sum(axis=(1, 2))))
    j = int(np.argmax(valid.sum(axis=(0, 2))))
    i = int(np.argmax(valid.sum(axis=(0, 1))))
    cuts = {"axial": (lambda v: v[k], k), "coronal": (lambda v: np.flipud(v[:, j, :]), j),
            "sagittal": (lambda v: np.flipud(v[:, :, i]), i)}

    fig = _figure(figsize=(17, 13))
    axes = fig.subplots(4, 3)
    for col, (view, (cut, idx)) in enumerate(cuts.items()):
        base = cut(raw)
        vmax = float(np.percentile(base[base > 0], 99)) if (base > 0).any() else 1
        axes[0][col].imshow(base, cmap="gray", vmax=vmax, interpolation="nearest")
        axes[0][col].set_title(f"{view} {idx}\nfirst-echo image (whole slice)", fontsize=9)
        axes[1][col].imshow(base, cmap="gray", vmax=vmax, interpolation="nearest")
        axes[1][col].imshow(np.ma.masked_where(~cut(valid), np.ones(base.shape)),
                            cmap="autumn", alpha=.45, interpolation="nearest")
        axes[1][col].set_title("red = voxels actually predicted", fontsize=9)
        axes[2][col].imshow(base, cmap="gray", vmax=vmax, interpolation="nearest")
        axes[2][col].imshow(np.ma.masked_where(~cut(segmask), np.ones(base.shape)),
                            cmap="winter", alpha=.45, interpolation="nearest")
        axes[2][col].set_title("blue = the 5 labelled segmentations", fontsize=9)
        p = cut(pred)
        axes[3][col].imshow(np.ma.masked_where(p == 0, p), cmap="tab10", vmin=0, vmax=10,
                            interpolation="nearest")
        axes[3][col].set_title("predicted class map", fontsize=9)
        for r in range(4):
            axes[r][col].set_axis_off()
    fig.suptitle("Pill2: what the 'full volume' prediction actually covers\n"
                 f"{int(valid.sum()):,} predicted of {np.prod(shape):,} voxels "
                 f"({100*valid.mean():.1f}%) — the rest is below the Otsu foreground threshold",
                 fontsize=13)
    _save(fig, OUT / "pill2_prediction_extent_check.png", [])
    print(f"\nsaved {OUT/'pill2_prediction_extent_check.png'}")


main()
