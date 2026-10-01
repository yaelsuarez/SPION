"""Do secondary transformations of I0_normalized improve cross-phantom agreement?

Starts from the existing Strategy-B I0_normalized. Nothing is refitted, no
existing result is modified, no classifier is trained.
"""
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT  # noqa: E402
import numpy as np
import pandas as pd

MRI = DATA_ROOT
DATA = MRI / "Normalized_data_i0_rs" / "Segmentations"
OUT = MRI / "i0_rs" / "model_diagnostics" / "normalization_comparison"
ORDER = ("0", "0.05", "0.1", "0.2", "0.25", "0.3", "water", "tissue")
SAMPLES = ("Syringes", "3D", "Pill1", "Pill2")
TRAINING = ("Syringes", "3D", "Pill1")          # Pill2 is held out
REPS = ("I0_normalized", "I0_zscore", "I0_rank", "I0_residual")
SEED = 42


def main():
    import echoviewer  # noqa: F401
    from echoviewer.diagnostics import (LABEL_COLORS, _figure, _save,
                                        bhattacharyya_overlap, subsample)

    OUT.mkdir(parents=True, exist_ok=True)
    saved = []

    # ---- load, same population and quality filter as before -------------
    rows = []
    for sample_dir in sorted(DATA.iterdir()):
        if not sample_dir.is_dir():
            continue
        for seg_dir in sorted(sample_dir.iterdir()):
            if not seg_dir.is_dir():
                continue
            block = pd.read_csv(seg_dir / "voxels.csv.gz",
                                usecols=["rs_selected", "I0_normalized", "flag_high_rs"])
            block = block[block.flag_high_rs == 0].drop(columns="flag_high_rs")
            block["sample"] = sample_dir.name
            block["material"] = seg_dir.name.split("_", 2)[2]   # folder name is authoritative
            rows.append(block)
    df = pd.concat(rows, ignore_index=True)
    del rows
    print(f"{len(df):,} voxels after the high-rs filter\n")

    # ---- secondary transformations --------------------------------------
    # 2. within-sample z-score
    grouped = df.groupby("sample")["I0_normalized"]
    df["I0_zscore"] = ((df.I0_normalized - grouped.transform("mean"))
                       / grouped.transform("std")).astype(np.float32)

    # 3. within-sample percentile rank, scaled to [0, 1]
    df["I0_rank"] = df.groupby("sample")["I0_normalized"].rank(pct=True).astype(np.float32)

    # 4. residual after removing a smooth dependence on rs, fitted on the
    #    training samples only and then applied unchanged to Pill2
    train = df[df["sample"].isin(TRAINING)]
    coefficients = np.polyfit(train.rs_selected.to_numpy(np.float64),
                              train.I0_normalized.to_numpy(np.float64), deg=2)
    trend = np.poly1d(coefficients)
    df["I0_residual"] = (df.I0_normalized - trend(df.rs_selected)).astype(np.float32)
    predicted = trend(train.rs_selected.to_numpy(np.float64))
    ss_res = float(np.sum((train.I0_normalized - predicted) ** 2))
    ss_tot = float(np.sum((train.I0_normalized - train.I0_normalized.mean()) ** 2))
    print(f"Residual trend f(rs) fitted on {TRAINING} only "
          f"({len(train):,} voxels), degree 2")
    print(f"  I0_normalized = {coefficients[0]:+.4g}*rs^2 {coefficients[1]:+.4g}*rs "
          f"{coefficients[2]:+.4g}   R^2 = {1 - ss_res/ss_tot:.4f}\n")
    del train

    def feats(rep, sample=None, material=None):
        mask = np.ones(len(df), dtype=bool)
        if sample:
            mask &= (df["sample"] == sample).to_numpy()
        if material:
            mask &= (df["material"] == material).to_numpy()
        return np.column_stack([df.rs_selected.to_numpy(np.float64)[mask],
                                df[rep].to_numpy(np.float64)[mask]])

    # ---- A. cross-phantom agreement, same material ----------------------
    pairs = []
    for material in ORDER:
        present = [s for s in SAMPLES
                   if ((df["sample"] == s) & (df["material"] == material)).any()]
        for a, b in combinations(present, 2):
            entry = {"material": material, "sample_A": a, "sample_B": b,
                     "n_A": int(((df["sample"] == a) & (df["material"] == material)).sum()),
                     "n_B": int(((df["sample"] == b) & (df["material"] == material)).sum())}
            for rep in REPS:
                entry[rep] = bhattacharyya_overlap(feats(rep, a, material),
                                                   feats(rep, b, material))
            pairs.append(entry)
    cross = pd.DataFrame(pairs)

    # ---- B. pooled between-material matrices ----------------------------
    materials = [m for m in ORDER if (df["material"] == m).any()]
    matrices = {}
    for rep in REPS:
        matrix = np.full((len(materials), len(materials)), np.nan)
        for i, a in enumerate(materials):
            fa = feats(rep, material=a)
            for j, b in enumerate(materials):
                matrix[i, j] = 1.0 if i == j else bhattacharyya_overlap(fa, feats(rep, material=b))
        matrices[rep] = matrix
        pd.DataFrame(matrix, index=materials, columns=materials).to_csv(
            OUT / f"overlap_matrix_{rep}.csv")

    # ---- print -----------------------------------------------------------
    print("=" * 96)
    print("HEADLINE — cross-phantom agreement (same material, different phantom). HIGHER is better.")
    print("           material separation = mean off-diagonal between materials. LOWER is better.")
    print("=" * 96)
    print(f"{'Representation':18s}{'Mean cross-phantom':>20s}{'Median cross-phantom':>22s}"
          f"{'Material separation':>21s}{'tissue vs 0.1':>15s}")
    summary = []
    for rep in REPS:
        off = matrices[rep][~np.eye(len(materials), dtype=bool)]
        t01 = matrices[rep][materials.index("tissue"), materials.index("0.1")]
        summary.append({"representation": rep,
                        "mean_cross_phantom": cross[rep].mean(),
                        "median_cross_phantom": cross[rep].median(),
                        "mean_between_material": np.nanmean(off),
                        "tissue_vs_0.1": t01})
        print(f"{rep:18s}{cross[rep].mean():>20.3f}{cross[rep].median():>22.3f}"
              f"{np.nanmean(off):>21.3f}{t01:>15.3f}")

    print("\n" + "=" * 96)
    print("FULL TABLE — same material, different phantom (higher = phantoms agree)")
    print("=" * 96)
    print(f"{'material':9s}{'A':10s}{'B':9s}{'n_A':>9s}{'n_B':>9s}"
          + "".join(f"{r.replace('I0_',''):>13s}" for r in REPS))
    print("-" * 96)
    for _, r in cross.iterrows():
        print(f"{r.material:9s}{r.sample_A:10s}{r.sample_B:9s}{r.n_A:>9,}{r.n_B:>9,}"
              + "".join(f"{r[rep]:>13.3f}" for rep in REPS))
    print("-" * 96)
    print(f"{'MEAN':46s}" + "".join(f"{cross[rep].mean():>13.3f}" for rep in REPS))
    print(f"{'MEDIAN':46s}" + "".join(f"{cross[rep].median():>13.3f}" for rep in REPS))

    print("\nBetween-material overlap matrices (pooled over phantoms, lower = better separated)")
    for rep in REPS:
        print(f"\n{rep}:")
        print(f"{'':9s}" + "".join(f"{m:>8s}" for m in materials))
        for i, a in enumerate(materials):
            print(f"{a:9s}" + "".join(f"{matrices[rep][i, j]:8.3f}" for j in range(len(materials))))

    cross.to_csv(OUT / "overlap_comparison.csv", index=False)
    pd.DataFrame(summary).to_csv(OUT / "overlap_summary.csv", index=False)

    # ---- figures ---------------------------------------------------------
    figure = _figure(figsize=(15, 6))
    axes = figure.subplots()
    x = np.arange(len(cross))
    width = 0.2
    colours = {"I0_normalized": "#4a6fe3", "I0_zscore": "#c9314e",
               "I0_rank": "#e8b647", "I0_residual": "#67c2a3"}
    for offset, rep in enumerate(REPS):
        axes.bar(x + (offset - 1.5) * width, cross[rep], width,
                 label=rep, color=colours[rep])
    axes.set_xticks(x)
    axes.set_xticklabels([f"{r.material}\n{r.sample_A[:4]}–{r.sample_B[:4]}"
                          for r in cross.itertuples()], fontsize=7)
    axes.set_ylabel("cross-phantom overlap (higher = phantoms agree)")
    axes.axhline(1.0, color="black", ls="--", lw=0.8)
    axes.set_title("Same material in different phantoms, by I0 representation\n"
                   "all start from Strategy-B I0_normalized; high-rs voxels excluded")
    axes.legend()
    axes.grid(alpha=0.25, axis="y")
    _save(figure, OUT / "overlap_comparison.png", saved)

    figure = _figure(figsize=(19, 5))
    panels = figure.subplots(1, 4)
    for axes, rep in zip(panels, REPS):
        image = axes.imshow(matrices[rep], cmap="RdYlGn_r", vmin=0, vmax=1)
        axes.set_xticks(range(len(materials)))
        axes.set_xticklabels(materials, rotation=45, ha="right", fontsize=7)
        axes.set_yticks(range(len(materials)))
        axes.set_yticklabels(materials, fontsize=7)
        for i in range(len(materials)):
            for j in range(len(materials)):
                axes.text(j, i, f"{matrices[rep][i, j]:.2f}", ha="center", va="center",
                          fontsize=6, color="white" if matrices[rep][i, j] > 0.6 else "black")
        axes.set_title(rep, fontsize=10)
        figure.colorbar(image, ax=axes, fraction=0.046)
    figure.suptitle("Between-material overlap (pooled over phantoms) — lower is better", fontsize=12)
    _save(figure, OUT / "overlap_matrices.png", saved)

    figure = _figure(figsize=(19, 5))
    panels = figure.subplots(1, 4)
    for axes, rep in zip(panels, REPS):
        rng = np.random.default_rng(SEED)
        for material in materials:
            mask = (df["material"] == material).to_numpy()
            x_values, y_values = subsample(
                [df.rs_selected.to_numpy(np.float32)[mask],
                 df[rep].to_numpy(np.float32)[mask]], 40_000, rng)
            axes.scatter(x_values, y_values, s=2, alpha=0.12, linewidths=0,
                         color=LABEL_COLORS[material], label=material)
        axes.set_xlabel("rs_selected (1/ms)")
        axes.set_ylabel(rep)
        axes.set_title(rep, fontsize=10)
        axes.grid(alpha=0.25)
    legend = panels[-1].legend(markerscale=6, fontsize=7)
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
    figure.suptitle(f"Feature space per representation — seed {SEED}, subsampled for display",
                    fontsize=12)
    _save(figure, OUT / "feature_space_comparison.png", saved)

    print(f"\nSaved to {OUT}")
    for path in saved:
        print(f"  {path.name}")


main()
