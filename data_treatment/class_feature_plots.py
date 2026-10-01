"""Per-sample rs x I0_residual scatter and per-class feature distributions. Read-only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT  # noqa: E402
import numpy as np, pandas as pd

MRI = DATA_ROOT
TRAIN = MRI / "Normalized_data_i0_rs" / "Segmentations_training"
PILL2 = MRI / "i0_rs" / "Volume_for_testing" / "Pill2_segmentation_complete" / "voxels_complete.csv.gz"
OUT = MRI / "Training" / "diagnosis" / "class"
CLASSES = ["0", "0.05", "0.1", "0.2", "0.25", "0.3", "tissue"]
FEATURES = [("rs_selected", "rs_selected (1/ms)"), ("I0_residual", "I0_residual")]
SEED, MAXN = 42, 60_000


def main():
    import echoviewer  # noqa: F401
    from echoviewer.diagnostics import LABEL_COLORS, _figure, _save, subsample, iqr

    OUT.mkdir(parents=True, exist_ok=True)
    frames = {}
    for s in ("Syringes", "3D", "Pill1"):
        parts = []
        for d in sorted(p for p in (TRAIN / s).iterdir() if p.is_dir()):
            m = d.name.split("_", 2)[2]
            if m not in CLASSES:
                continue
            b = pd.read_csv(d / "voxels.csv.gz", usecols=["rs_selected", "I0_residual"])
            b["material"] = m
            parts.append(b)
        frames[s] = pd.concat(parts, ignore_index=True)
    p2 = pd.read_csv(PILL2, usecols=["rs_selected", "I0_residual", "material"])
    p2["material"] = p2.material.astype(str)
    frames["Pill2"] = p2[p2.material.isin(CLASSES)].reset_index(drop=True)
    frames["All samples"] = pd.concat(
        [f.assign(sample=k) for k, f in frames.items()], ignore_index=True)

    everything = frames["All samples"]
    xlim = (0, float(np.percentile(everything.rs_selected, 99.8)) * 1.05)
    ylim = (float(np.percentile(everything.I0_residual, 0.2)) * 1.05,
            float(np.percentile(everything.I0_residual, 99.8)) * 1.05)
    lims = {"rs_selected": xlim, "I0_residual": ylim}

    stats = []
    for name, df in frames.items():
        present = [c for c in CLASSES if (df.material == c).any()]
        fig = _figure(figsize=(19, 5.8))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1, 1], wspace=.25)

        ax = fig.add_subplot(gs[0, 0])
        rng = np.random.default_rng(SEED)
        total = drawn = 0
        for c in present:
            sub = df[df.material == c]
            total += len(sub)
            x, y = subsample([sub.rs_selected.to_numpy(np.float32),
                              sub.I0_residual.to_numpy(np.float32)], MAXN, rng)
            drawn += x.size
            ax.scatter(x, y, s=3, alpha=.14, linewidths=0, color=LABEL_COLORS[c],
                       label=f"{c} (n={len(sub):,})")
        ax.axhline(0, color="black", ls="--", lw=.8)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.grid(alpha=.25)
        ax.set_xlabel("rs_selected (1/ms)"); ax.set_ylabel("I0_residual")
        ax.set_title(f"rs_selected x I0_residual\n{drawn:,} of {total:,} plotted, seed {SEED}",
                     fontsize=10)
        lg = ax.legend(markerscale=5, fontsize=8)
        for h in lg.legend_handles: h.set_alpha(1.0)

        for col, (feat, label) in enumerate(FEATURES, start=1):
            ax = fig.add_subplot(gs[0, col])
            data = [df[df.material == c][feat].to_numpy(np.float64) for c in present]
            parts = ax.violinplot(data, showextrema=False, widths=.85)
            for body, c in zip(parts["bodies"], present):
                body.set_facecolor(LABEL_COLORS[c]); body.set_alpha(.7)
            for pos, (c, v) in enumerate(zip(present, data), start=1):
                q1, q3 = iqr(v)
                ax.vlines(pos, q1, q3, color="black", lw=3.5, zorder=3)
                ax.plot(pos, np.median(v), "o", color="white", markeredgecolor="black",
                        markersize=5, zorder=4)
                stats.append({"sample": name, "material": c, "feature": feat,
                              "n": len(v), "median": float(np.median(v)),
                              "q1": q1, "q3": q3, "iqr": q3 - q1,
                              "p05": float(np.percentile(v, 5)),
                              "p95": float(np.percentile(v, 95))})
            ax.set_xticks(range(1, len(present) + 1)); ax.set_xticklabels(present, rotation=30)
            ax.set_ylabel(label); ax.set_ylim(*lims[feat]); ax.grid(alpha=.25, axis="y")
            ax.set_title(f"{feat} by class\n(all voxels; median + IQR marked)", fontsize=10)
            if feat == "I0_residual":
                ax.axhline(0, color="black", ls="--", lw=.8)

        fig.suptitle(f"{name} — {len(df):,} voxels", fontsize=14)
        tag = name.replace(" ", "_")
        _save(fig, OUT / f"class_features_{tag}.png", [])
        print(f"  {name:12s} {len(df):>9,} voxels, classes {present}")

    t = pd.DataFrame(stats)
    t.round(5).to_csv(OUT / "class_feature_stats.csv", index=False)
    print(f"\nsaved {len(list(OUT.iterdir()))} files to {OUT}")
    piv = t[t.feature == "I0_residual"].pivot_table(index="material", columns="sample",
                                                    values="median")
    print("\nmedian I0_residual by class and sample:")
    print(piv.reindex([c for c in CLASSES if c in piv.index]).round(3).to_string())

main()
