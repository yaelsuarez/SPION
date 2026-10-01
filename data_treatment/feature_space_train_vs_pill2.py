"""Training vs complete-Pill2 distributions in rs_selected x I0_residual. Read-only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT  # noqa: E402
import numpy as np, pandas as pd

MRI = DATA_ROOT
TRAIN = MRI / "Normalized_data_i0_rs" / "Segmentations_training"
PILL2 = MRI / "i0_rs" / "Volume_for_testing" / "Pill2_segmentation_complete" / "voxels_complete.csv.gz"
OUT = MRI / "Training" / "diagnosis"
CLASSES = ["0", "0.05", "0.1", "0.2", "0.25", "0.3", "tissue"]
SEED, MAXN = 42, 60_000


def main():
    import echoviewer  # noqa: F401
    from echoviewer.diagnostics import LABEL_COLORS, _figure, _save, subsample

    tr = []
    for s in ("Syringes", "3D", "Pill1"):
        for d in sorted(p for p in (TRAIN / s).iterdir() if p.is_dir()):
            mat = d.name.split("_", 2)[2]
            if mat not in CLASSES:
                continue
            b = pd.read_csv(d / "voxels.csv.gz", usecols=["rs_selected", "I0_residual"])
            b["material"] = mat; b["sample"] = s
            tr.append(b)
    tr = pd.concat(tr, ignore_index=True)
    p2 = pd.read_csv(PILL2, usecols=["rs_selected", "I0_residual", "material"])
    p2["material"] = p2.material.astype(str)
    p2 = p2[p2.material.isin(CLASSES)]
    print(f"training {len(tr):,} voxels  |  Pill2 complete {len(p2):,} voxels")

    both = pd.concat([tr[["rs_selected", "I0_residual"]], p2[["rs_selected", "I0_residual"]]])
    xlim = (0, float(np.percentile(both.rs_selected, 99.8)) * 1.05)
    ylim = (float(np.percentile(both.I0_residual, 0.2)) * 1.05,
            float(np.percentile(both.I0_residual, 99.8)) * 1.05)
    del both

    def draw(ax, df, title, alpha=0.13):
        rng = np.random.default_rng(SEED)
        total = drawn = 0
        for m in CLASSES:
            sub = df[df.material == m]
            if not len(sub):
                continue
            total += len(sub)
            x, y = subsample([sub.rs_selected.to_numpy(np.float32),
                              sub.I0_residual.to_numpy(np.float32)], MAXN, rng)
            drawn += x.size
            ax.scatter(x, y, s=3, alpha=alpha, linewidths=0, color=LABEL_COLORS[m],
                       label=f"{m} (n={len(sub):,})")
        ax.axhline(0, color="black", ls="--", lw=.8)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.grid(alpha=.25)
        ax.set_xlabel("rs_selected (1/ms)"); ax.set_ylabel("I0_residual")
        ax.set_title(f"{title}\n{drawn:,} of {total:,} plotted, seed {SEED}", fontsize=10)
        return total, drawn

    shared = [m for m in CLASSES if (tr.material == m).any() and (p2.material == m).any()]
    fig = _figure(figsize=(19, 12))
    gs = fig.add_gridspec(2, len(shared), height_ratios=[1.5, 1], hspace=.3, wspace=.25)

    ax = fig.add_subplot(gs[0, 0:2]); draw(ax, tr, "TRAINING  (Syringes + 3D + Pill1)")
    lg = ax.legend(markerscale=5, fontsize=8)
    for h in lg.legend_handles: h.set_alpha(1.0)

    ax = fig.add_subplot(gs[0, 2:4]); draw(ax, p2, "PILL2  (complete segmentation, 384,315 voxels)")
    lg = ax.legend(markerscale=5, fontsize=8)
    for h in lg.legend_handles: h.set_alpha(1.0)

    ax = fig.add_subplot(gs[0, 4:])
    rng = np.random.default_rng(SEED)
    x, y = subsample([tr.rs_selected.to_numpy(np.float32),
                      tr.I0_residual.to_numpy(np.float32)], 150_000, rng)
    ax.scatter(x, y, s=3, alpha=.08, linewidths=0, color="#b0b0b0", label="training (all)")
    for m in CLASSES:
        sub = p2[p2.material == m]
        if not len(sub): continue
        x, y = subsample([sub.rs_selected.to_numpy(np.float32),
                          sub.I0_residual.to_numpy(np.float32)], MAXN, rng)
        ax.scatter(x, y, s=3, alpha=.15, linewidths=0, color=LABEL_COLORS[m], label=f"Pill2 {m}")
    ax.axhline(0, color="black", ls="--", lw=.8)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.grid(alpha=.25)
    ax.set_xlabel("rs_selected (1/ms)"); ax.set_ylabel("I0_residual")
    ax.set_title("OVERLAY — Pill2 on the grey training cloud", fontsize=10)
    lg = ax.legend(markerscale=5, fontsize=7)
    for h in lg.legend_handles: h.set_alpha(1.0)

    for col, m in enumerate(shared):
        ax = fig.add_subplot(gs[1, col])
        rng = np.random.default_rng(SEED)
        for df, lab, c, mk in ((tr[tr.material == m], "training", "#8f8f8f", "o"),
                               (p2[p2.material == m], "Pill2", LABEL_COLORS[m], "o")):
            x, y = subsample([df.rs_selected.to_numpy(np.float32),
                              df.I0_residual.to_numpy(np.float32)], 30_000, rng)
            ax.scatter(x, y, s=3, alpha=.18, linewidths=0, color=c, marker=mk,
                       label=f"{lab} (n={len(df):,})")
        ax.axhline(0, color="black", ls="--", lw=.6)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.grid(alpha=.25)
        ax.set_title(f"material {m}", fontsize=10)
        ax.set_xlabel("rs_selected"); 
        if col == 0: ax.set_ylabel("I0_residual")
        lg = ax.legend(markerscale=5, fontsize=7)
        for h in lg.legend_handles: h.set_alpha(1.0)

    fig.suptitle("rs_selected x I0_residual — training vs complete Pill2 segmentation "
                 "(label-light normalisation, identical f(rs))", fontsize=15)
    _save(fig, OUT / "feature_space_training_vs_pill2.png", [])

    # median shift per shared class, for the caption/table
    rows = []
    for m in shared:
        a, b = tr[tr.material == m], p2[p2.material == m]
        rows.append({"material": m, "n_train": len(a), "n_pill2": len(b),
                     "train_rs_med": a.rs_selected.median(), "pill2_rs_med": b.rs_selected.median(),
                     "train_res_med": a.I0_residual.median(), "pill2_res_med": b.I0_residual.median()})
    t = pd.DataFrame(rows)
    t["d_rs"] = t.pill2_rs_med - t.train_rs_med
    t["d_residual"] = t.pill2_res_med - t.train_res_med
    t.round(4).to_csv(OUT / "feature_space_train_vs_pill2_shift.csv", index=False)
    print(t.round(4).to_string(index=False))
    print("saved", OUT / "feature_space_training_vs_pill2.png")

main()
