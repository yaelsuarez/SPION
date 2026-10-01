"""Does B_selected add class separation? Read-only diagnostic, training vs Pill2."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT  # noqa: E402
import numpy as np, pandas as pd

MRI = DATA_ROOT
TRAIN = MRI / "Normalized_data_i0_rs" / "Segmentations_training"
PILL2 = MRI / "i0_rs" / "Volume_for_testing" / "Pill2_segmentation_complete" / "voxels_complete.csv.gz"
OUT = MRI / "Training" / "diagnosis" / "B_parameter"
CLASSES = ["0", "0.05", "0.1", "0.2", "0.25", "0.3", "tissue"]
PAIRS = [("rs_selected", "I0_residual"), ("rs_selected", "B_selected"),
         ("I0_residual", "B_selected")]
AXIS = {"rs_selected": "rs_selected (1/ms)", "I0_residual": "I0_residual",
        "B_selected": "B_selected (a.u.)"}
SEED, MAXN = 42, 60_000


def main():
    import echoviewer  # noqa: F401
    from echoviewer.diagnostics import (LABEL_COLORS, _figure, _save, subsample,
                                        bhattacharyya_overlap, iqr)

    OUT.mkdir(parents=True, exist_ok=True)
    cols = ["rs_selected", "I0_residual", "B_selected"]

    parts = []
    for s in ("Syringes", "3D", "Pill1"):
        for d in sorted(p for p in (TRAIN / s).iterdir() if p.is_dir()):
            m = d.name.split("_", 2)[2]
            if m not in CLASSES:
                continue
            b = pd.read_csv(d / "voxels.csv.gz", usecols=cols)
            b["material"] = m
            parts.append(b)
    train = pd.concat(parts, ignore_index=True); del parts
    p2 = pd.read_csv(PILL2, usecols=cols + ["material"])
    p2["material"] = p2.material.astype(str)
    p2 = p2[p2.material.isin(CLASSES)].reset_index(drop=True)
    data = {"Training (Syringes + 3D + Pill1)": train, "Pill2 (segmented)": p2}
    print(f"training {len(train):,}  |  Pill2 {len(p2):,}")

    both = pd.concat([train[cols], p2[cols]])
    lim = {c: (float(np.percentile(both[c], 0.2)), float(np.percentile(both[c], 99.8)))
           for c in cols}
    lim["rs_selected"] = (0, lim["rs_selected"][1] * 1.05)
    lim["B_selected"] = (0, lim["B_selected"][1] * 1.05)
    del both

    def scatter(ax, df, xc, yc):
        rng = np.random.default_rng(SEED)
        total = drawn = 0
        for c in CLASSES:
            sub = df[df.material == c]
            if not len(sub):
                continue
            total += len(sub)
            x, y = subsample([sub[xc].to_numpy(np.float32), sub[yc].to_numpy(np.float32)],
                             MAXN, rng)
            drawn += x.size
            ax.scatter(x, y, s=3, alpha=.12, linewidths=0, color=LABEL_COLORS[c],
                       label=f"{c} (n={len(sub):,})")
        ax.set_xlim(*lim[xc]); ax.set_ylim(*lim[yc]); ax.grid(alpha=.25)
        ax.set_xlabel(AXIS[xc]); ax.set_ylabel(AXIS[yc])
        return total, drawn

    # combined 2 x 3
    fig = _figure(figsize=(19, 11))
    axes = fig.subplots(2, 3)
    for row, (name, df) in enumerate(data.items()):
        for col, (xc, yc) in enumerate(PAIRS):
            t, d = scatter(axes[row][col], df, xc, yc)
            axes[row][col].set_title(f"{name}\n{yc} vs {xc} — {d:,} of {t:,} plotted",
                                     fontsize=9)
    lg = axes[0][0].legend(markerscale=5, fontsize=7)
    for h in lg.legend_handles: h.set_alpha(1.0)
    fig.suptitle("Feature relationships, identical axes between training and Pill2 "
                 f"(seed {SEED}; subsampling affects display only)", fontsize=14)
    _save(fig, OUT / "feature_pairs_training_vs_pill2.png", [])

    for xc, yc in PAIRS:
        fig = _figure(figsize=(14, 6.2))
        axes = fig.subplots(1, 2)
        for ax, (name, df) in zip(axes, data.items()):
            t, d = scatter(ax, df, xc, yc)
            ax.set_title(f"{name}\n{d:,} of {t:,} plotted, seed {SEED}", fontsize=10)
            lg = ax.legend(markerscale=5, fontsize=7)
            for h in lg.legend_handles: h.set_alpha(1.0)
        fig.suptitle(f"{yc} vs {xc}", fontsize=14)
        _save(fig, OUT / f"{yc}_vs_{xc}.png", [])

    # ---- does B add separation? -----------------------------------------
    sets = {"rs": ["rs_selected"],
            "rs + I0_residual": ["rs_selected", "I0_residual"],
            "rs + B": ["rs_selected", "B_selected"],
            "rs + I0_residual + B": ["rs_selected", "I0_residual", "B_selected"]}
    focus = [("0.2", "0.25"), ("tissue", "0.05"), ("tissue", "0.1"),
             ("0.3", "0"), ("0.3", "0.1"), ("0.3", "0.2"), ("0.3", "0.25")]
    rows = []
    for name, df in data.items():
        for a, b in focus:
            if not ((df.material == a).any() and (df.material == b).any()):
                rows.append({"dataset": name, "class_a": a, "class_b": b,
                             **{k: np.nan for k in sets}, "note": "class absent"})
                continue
            entry = {"dataset": name, "class_a": a, "class_b": b, "note": ""}
            for label, fc in sets.items():
                entry[label] = bhattacharyya_overlap(
                    df[df.material == a][fc].to_numpy(np.float64),
                    df[df.material == b][fc].to_numpy(np.float64))
            rows.append(entry)
    ov = pd.DataFrame(rows)
    ov.round(4).to_csv(OUT / "overlap_with_without_B.csv", index=False)
    print("\nBhattacharyya overlap (lower = better separated)")
    print(ov.round(3).to_string(index=False))

    bstats = []
    for name, df in data.items():
        for c in CLASSES:
            sub = df[df.material == c]
            if not len(sub):
                continue
            q1, q3 = iqr(sub.B_selected.to_numpy())
            bstats.append({"dataset": name, "material": c, "n": len(sub),
                           "pct_B_zero": 100 * float((sub.B_selected == 0).mean()),
                           "median_B": float(sub.B_selected.median()), "q1": q1, "q3": q3})
    bs = pd.DataFrame(bstats)
    bs.round(3).to_csv(OUT / "B_by_class.csv", index=False)
    print("\nB_selected by class")
    print(bs.round(2).to_string(index=False))

    # ---- summary ---------------------------------------------------------
    def delta(dataset, a, b):
        r = ov[(ov.dataset == dataset) & (ov.class_a == a) & (ov.class_b == b)]
        if not len(r) or pd.isna(r["rs + I0_residual"].iloc[0]):
            return None
        return (float(r["rs"].iloc[0]), float(r["rs + I0_residual"].iloc[0]),
                float(r["rs + B"].iloc[0]), float(r["rs + I0_residual + B"].iloc[0]))

    tr_name, p2_name = list(data)
    lines = ["# Does B_selected add class separation?", "",
             "Read-only diagnostic. Nothing fitted, retrained, normalised or modified; "
             "Pill2 was not used to tune anything.", "",
             f"Training {len(train):,} voxels | Pill2 segmented {len(p2):,} voxels. "
             f"Bhattacharyya overlap, complete populations; 0 = separated, 1 = identical.", "",
             "| comparison | dataset | rs | rs+I0res | rs+B | rs+I0res+B |",
             "|---|---|---|---|---|---|"]
    for a, b in focus:
        for name in data:
            d = delta(name, a, b)
            if d is None:
                lines.append(f"| {a} vs {b} | {name.split()[0]} | – | – | – | "
                             f"not present in this dataset |")
            else:
                lines.append(f"| {a} vs {b} | {name.split()[0]} | {d[0]:.3f} | {d[1]:.3f} | "
                             f"{d[2]:.3f} | {d[3]:.3f} |")
    lines += ["", "## B_selected by class", "",
              "| dataset | class | n | % B exactly 0 | median B | IQR |", "|---|---|---|---|---|---|"]
    for _, r in bs.iterrows():
        lines.append(f"| {r.dataset.split()[0]} | {r.material} | {int(r.n):,} | "
                     f"{r.pct_B_zero:.1f}% | {r.median_B:.1f} | {r.q1:.1f}–{r.q3:.1f} |")
    (OUT / "summary.md").write_text("\n".join(lines))
    print(f"\nsaved {len(list(OUT.iterdir()))} files to {OUT}")


main()
