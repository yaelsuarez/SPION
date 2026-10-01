"""Final feature-space check: rs_selected vs I0_residual, per phantom and combined."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT  # noqa: E402
import numpy as np, pandas as pd

MRI = DATA_ROOT
DATA = MRI / "Normalized_data_i0_rs" / "Segmentations"
OUT = MRI / "i0_rs" / "model_diagnostics" / "normalization_comparison"
ORDER = ("0", "0.05", "0.1", "0.2", "0.25", "0.3", "water", "tissue")
SAMPLES = ("Syringes", "3D", "Pill1", "Pill2")
TRAINING = ("Syringes", "3D", "Pill1")
SEED, MAX_PER_GROUP = 42, 40_000

def main():
    import echoviewer  # noqa: F401
    from echoviewer.diagnostics import LABEL_COLORS, _figure, _save, subsample

    parts = []
    for sd in sorted(DATA.iterdir()):
        if not sd.is_dir(): continue
        for gd in sorted(sd.iterdir()):
            if not gd.is_dir(): continue
            b = pd.read_csv(gd/"voxels.csv.gz",
                            usecols=["rs_selected","I0_normalized","flag_high_rs"])
            b = b[b.flag_high_rs == 0].drop(columns="flag_high_rs")
            b["sample"] = sd.name; b["material"] = gd.name.split("_",2)[2]
            parts.append(b)
    df = pd.concat(parts, ignore_index=True); del parts

    # identical residual definition to the previous comparison
    tr = df[df["sample"].isin(TRAINING)]
    c = np.polyfit(tr.rs_selected.to_numpy(np.float64),
                   tr.I0_normalized.to_numpy(np.float64), 2)
    f = np.poly1d(c)
    df["I0_residual"] = (df.I0_normalized - f(df.rs_selected)).astype(np.float32)
    print(f"f(rs) = {c[0]:+.4g}*rs^2 {c[1]:+.4g}*rs {c[2]:+.4g}   "
          f"(fitted on {TRAINING}, {len(tr):,} voxels)")
    del tr

    materials = [m for m in ORDER if (df["material"] == m).any()]
    lim_x = (0, float(np.percentile(df.rs_selected, 99.9))*1.05)
    lo, hi = np.percentile(df.I0_residual, [0.2, 99.8])
    lim_y = (float(lo)*1.05, float(hi)*1.05)

    def draw(ax, sample=None):
        rng = np.random.default_rng(SEED)
        total = drawn = 0
        for m in materials:
            mask = (df["material"] == m).to_numpy()
            if sample: mask &= (df["sample"] == sample).to_numpy()
            if not mask.any(): continue
            total += int(mask.sum())
            x, y = subsample([df.rs_selected.to_numpy(np.float32)[mask],
                              df.I0_residual.to_numpy(np.float32)[mask]], MAX_PER_GROUP, rng)
            drawn += x.size
            ax.scatter(x, y, s=3, alpha=0.15, linewidths=0,
                       color=LABEL_COLORS[m], label=f"{m} (n={int(mask.sum()):,})")
        ax.axhline(0, color="black", ls="--", lw=0.9)
        ax.set_xlabel("rs_selected (1/ms)"); ax.set_ylabel("I0_residual")
        ax.set_xlim(*lim_x); ax.set_ylim(*lim_y); ax.grid(alpha=.25)
        return total, drawn

    fig = _figure(figsize=(21, 12))
    gs = fig.add_gridspec(2, 3, hspace=.25, wspace=.2)
    report = []
    for i, s in enumerate(SAMPLES):
        ax = fig.add_subplot(gs[i//2, i%2] if i < 2 else gs[(i)//2, (i)%2])
        t, d = draw(ax, s); report.append((s, t, d))
        ax.set_title(f"{s} — {t:,} voxels, {d:,} plotted", fontsize=11)
        lg = ax.legend(markerscale=5, fontsize=7)
        for h in lg.legend_handles: h.set_alpha(1.0)
    ax = fig.add_subplot(gs[:, 2])
    t, d = draw(ax); report.append(("all samples", t, d))
    ax.set_title(f"All samples — {t:,} voxels, {d:,} plotted", fontsize=12)
    lg = ax.legend(markerscale=5, fontsize=8)
    for h in lg.legend_handles: h.set_alpha(1.0)
    fig.suptitle("rs_selected vs I0_residual  (I0_residual = Strategy-B I0_normalized − f(rs), "
                 f"f fitted on Syringes+3D+Pill1)\nidentical axes, seed {SEED}", fontsize=13)
    _save(fig, OUT/"feature_space_rs_vs_I0_residual.png", [])

    print(f"\n{'panel':14s}{'total':>12s}{'plotted':>10s}{'fraction':>10s}  seed")
    for n,t,d in report: print(f"{n:14s}{t:>12,}{d:>10,}{100*d/t:>9.1f}%  {SEED}")

    print("\nmedian I0_residual by sample and material")
    piv = df.groupby(["material","sample"]).I0_residual.median().unstack()
    piv = piv.reindex(index=[m for m in ORDER if m in piv.index], columns=list(SAMPLES))
    print(piv.round(3).to_string())
    print("\nspread across phantoms (max-min of the medians above):")
    print((piv.max(axis=1)-piv.min(axis=1)).round(3).to_string())
    print("\nmedian I0_residual by material (pooled) vs what f(rs) removes at that rs:")
    for m in materials:
        sub = df[df.material == m]
        print(f"  {m:7s} median residual {sub.I0_residual.median():+7.3f}   "
              f"median rs {sub.rs_selected.median():.4f}   f(rs) = {f(sub.rs_selected.median()):.3f}")
    print(f"\nSaved: {OUT/'feature_space_rs_vs_I0_residual.png'}")

main()
