"""READ-ONLY: does normalising B_selected make it phantom-invariant?

B_normalized = B_selected / reference_sample, using the SAME label-light reference
factors already derived for I0. B and I0 are both amplitudes of the same fitted
curve and share units, so the phantom intensity scale that divides one divides the
other. No new label is consulted; in particular no Pill2 concentration label is
used to derive or tune anything. Pill2 labels appear only in the final comparison.

A 0-SPION median of B cannot serve as a reference: class 0 is 62% (training) to
89% (Pill2) exactly zero, because those voxels select Model A.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT  # noqa: E402
import numpy as np, pandas as pd

MRI = DATA_ROOT
TRAIN = MRI / "Normalized_data_i0_rs" / "Segmentations_training"
PILL2 = MRI / "i0_rs" / "Volume_for_testing" / "Pill2_segmentation_complete" / "voxels_complete.csv.gz"
OUT = MRI / "Training" / "diagnosis" / "B_parameter" / "normalization_transfer"

# label-light reference factors, taken unchanged from the I0 normalisation
FACTORS = {"Syringes": 7301.30, "3D": 1069.995, "Pill1": 1294.1917, "Pill2": 2042.6199951171875}
CLASSES = ["0", "0.05", "0.1", "0.2", "0.25", "0.3", "tissue"]
FOCUS = ["0", "0.1", "0.25", "0.3", "tissue"]
PAIRS = [("0.2", "0.25"), ("tissue", "0.05"), ("tissue", "0.1"), ("0.3", "0.25")]
SEED = 42


def main():
    import echoviewer  # noqa: F401
    from echoviewer.diagnostics import (LABEL_COLORS, _figure, _save,
                                        bhattacharyya_overlap, overlap_1d, iqr)

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in ("Syringes", "3D", "Pill1"):
        for d in sorted(p for p in (TRAIN / s).iterdir() if p.is_dir()):
            m = d.name.split("_", 2)[2]
            if m not in CLASSES:
                continue
            b = pd.read_csv(d / "voxels.csv.gz", usecols=["B_selected"])
            b["material"] = m; b["sample"] = s
            rows.append(b)
    df = pd.concat(rows, ignore_index=True); del rows
    p2 = pd.read_csv(PILL2, usecols=["B_selected", "material"])
    p2["material"] = p2.material.astype(str); p2["sample"] = "Pill2"
    p2 = p2[p2.material.isin(CLASSES)]
    df = pd.concat([df, p2], ignore_index=True); del p2
    df["B_raw"] = df.B_selected
    df["B_normalized"] = df.B_selected / df["sample"].map(FACTORS)
    print(f"{len(df):,} voxels; factors {FACTORS}")

    # ---- per class x phantom statistics ---------------------------------
    stats = []
    for (samp, mat), g in df.groupby(["sample", "material"]):
        for feat in ("B_raw", "B_normalized"):
            v = g[feat].to_numpy(np.float64)
            q1, q3 = iqr(v)
            stats.append({"sample": samp, "material": mat, "feature": feat, "n": len(v),
                          "median": float(np.median(v)), "q1": q1, "q3": q3, "iqr": q3 - q1,
                          "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)),
                          "pct_zero": 100 * float((v == 0).mean())})
    st = pd.DataFrame(stats)
    st.round(6).to_csv(OUT / "class_phantom_statistics.csv", index=False)

    print("\nMedian B by class and phantom")
    for feat in ("B_raw", "B_normalized"):
        piv = st[st.feature == feat].pivot_table(index="material", columns="sample",
                                                 values="median")
        piv = piv.reindex([c for c in CLASSES if c in piv.index])
        print(f"\n  {feat}:")
        print(piv.round(5).to_string())

    # ---- same-class transfer: training (pooled) vs Pill2 -----------------
    print("\nSame-class transfer, training(pooled) vs Pill2  (overlap: higher = agrees)")
    trans = []
    tr = df[df["sample"] != "Pill2"]
    p2 = df[df["sample"] == "Pill2"]
    for m in FOCUS:
        a, b = tr[tr.material == m], p2[p2.material == m]
        if not len(a) or not len(b):
            continue
        e = {"material": m, "n_train": len(a), "n_pill2": len(b)}
        for feat in ("B_raw", "B_normalized"):
            e[f"{feat}_hist_overlap"] = overlap_1d(a[feat].to_numpy(np.float64),
                                                   b[feat].to_numpy(np.float64))
            e[f"{feat}_bhatt"] = bhattacharyya_overlap(a[feat].to_numpy(np.float64),
                                                       b[feat].to_numpy(np.float64))
            e[f"{feat}_median_shift"] = float(np.median(b[feat]) - np.median(a[feat]))
            mt = float(np.median(a[feat]))
            e[f"{feat}_relative_shift"] = (float(np.median(b[feat])) / mt - 1) if mt else np.nan
        trans.append(e)
    tt = pd.DataFrame(trans)
    print(tt[["material", "B_raw_hist_overlap", "B_normalized_hist_overlap",
              "B_raw_relative_shift", "B_normalized_relative_shift"]].round(4).to_string(index=False))

    # pairwise between individual phantoms, same class
    pw = []
    samples = ["Syringes", "3D", "Pill1", "Pill2"]
    for m in CLASSES:
        present = [s for s in samples if len(df[(df["sample"] == s) & (df.material == m)])]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                a = df[(df["sample"] == present[i]) & (df.material == m)]
                b = df[(df["sample"] == present[j]) & (df.material == m)]
                e = {"material": m, "sample_a": present[i], "sample_b": present[j]}
                for feat in ("B_raw", "B_normalized"):
                    e[feat] = overlap_1d(a[feat].to_numpy(np.float64),
                                         b[feat].to_numpy(np.float64))
                e["change"] = e["B_normalized"] - e["B_raw"]
                pw.append(e)
    pwd = pd.DataFrame(pw)
    pwd.round(4).to_csv(OUT / "same_class_cross_phantom_overlap.csv", index=False)
    tt.round(5).to_csv(OUT / "training_vs_pill2_transfer.csv", index=False)
    print(f"\nPairwise same-class agreement across phantoms (higher = agree): "
          f"raw mean {pwd.B_raw.mean():.3f}, normalized mean {pwd.B_normalized.mean():.3f}, "
          f"improved {int((pwd.change > 0).sum())}/{len(pwd)} pairs")

    # ---- class-pair separation, pooled training vs Pill2 ------------------
    sep = []
    for label, sub in (("training pooled", tr), ("Pill2", p2)):
        for a, b in PAIRS:
            if not len(sub[sub.material == a]) or not len(sub[sub.material == b]):
                sep.append({"dataset": label, "pair": f"{a} vs {b}",
                            "B_raw": np.nan, "B_normalized": np.nan, "note": "class absent"})
                continue
            e = {"dataset": label, "pair": f"{a} vs {b}", "note": ""}
            for feat in ("B_raw", "B_normalized"):
                e[feat] = overlap_1d(sub[sub.material == a][feat].to_numpy(np.float64),
                                     sub[sub.material == b][feat].to_numpy(np.float64))
            sep.append(e)
    sp = pd.DataFrame(sep)
    sp.round(4).to_csv(OUT / "class_pair_separation.csv", index=False)
    print("\nClass-pair overlap (lower = better separated)")
    print(sp.round(4).to_string(index=False))

    # ---- figures ---------------------------------------------------------
    for feat, tag in (("B_raw", "raw"), ("B_normalized", "normalized")):
        fig = _figure(figsize=(17, 5.5))
        axes = fig.subplots(1, 4)
        hi = float(np.percentile(df[feat], 99.5))
        for ax, s in zip(axes, samples):
            sub = df[df["sample"] == s]
            present = [c for c in CLASSES if (sub.material == c).any()]
            data = [np.clip(sub[sub.material == c][feat].to_numpy(np.float64), 0, hi)
                    for c in present]
            parts = ax.violinplot(data, showextrema=False, widths=.85)
            for body, c in zip(parts["bodies"], present):
                body.set_facecolor(LABEL_COLORS[c]); body.set_alpha(.7)
            for pos, v in enumerate(data, start=1):
                q1, q3 = iqr(v)
                ax.vlines(pos, q1, q3, color="black", lw=3)
                ax.plot(pos, np.median(v), "o", color="white", markeredgecolor="black", ms=5)
            ax.set_xticks(range(1, len(present) + 1)); ax.set_xticklabels(present, rotation=30)
            ax.set_ylim(0, hi); ax.grid(alpha=.25, axis="y")
            ax.set_title(f"{s} (n={len(sub):,})", fontsize=10)
            if s == samples[0]:
                ax.set_ylabel(feat)
        fig.suptitle(f"{feat} by class and phantom — identical y axis "
                     f"(clipped at the 99.5th percentile)", fontsize=13)
        _save(fig, OUT / f"B_{tag}_by_class_and_phantom.png", [])

    fig = _figure(figsize=(15, 6))
    axes = fig.subplots(1, 2)
    for ax, feat in zip(axes, ("B_raw", "B_normalized")):
        hi = float(np.percentile(df[feat], 99.5))
        for m in FOCUS:
            for sub, ls, lab in ((tr, "-", "training"), (p2, "--", "Pill2")):
                v = sub[sub.material == m][feat].to_numpy(np.float64)
                if not len(v):
                    continue
                ax.hist(np.clip(v, 0, hi), bins=120, range=(0, hi), density=True,
                        histtype="step", lw=1.5, ls=ls, color=LABEL_COLORS[m],
                        label=f"{m} {lab}")
        ax.set_xlabel(feat); ax.set_ylabel("density"); ax.set_yscale("log")
        ax.grid(alpha=.25); ax.set_title(feat, fontsize=11)
    axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle("Training (solid) vs Pill2 (dashed) — does normalisation align them?",
                 fontsize=13)
    _save(fig, OUT / "B_raw_vs_normalized_transfer.png", [])

    # ---- summary ---------------------------------------------------------
    lines = ["# Does normalising B improve phantom transfer?", "",
             "Read-only. Nothing fitted, trained or modified.", "",
             "`B_normalized = B_selected / reference_sample`, using the label-light I0 "
             "reference factors unchanged: " +
             ", ".join(f"{k} {v:g}" for k, v in FACTORS.items()) + ".", "",
             "B and I0 are amplitudes of the same fitted curve and share units, so the same "
             "phantom intensity scale applies. A 0-SPION median of B is unusable as a "
             "reference because class `0` is 62% (training) to 89% (Pill2) exactly zero — "
             "those voxels select Model A, where B is 0 by construction.", "",
             "## Same class, training pooled vs Pill2", "",
             "| class | raw overlap | normalized overlap | raw rel. shift | norm rel. shift |",
             "|---|---|---|---|---|"]
    for _, r in tt.iterrows():
        lines.append(f"| {r.material} | {r.B_raw_hist_overlap:.3f} | "
                     f"{r.B_normalized_hist_overlap:.3f} | "
                     f"{r.B_raw_relative_shift:+.3f} | {r.B_normalized_relative_shift:+.3f} |")
    lines += ["", "## Class-pair separation (lower = better)", "",
              "| dataset | pair | raw B | normalized B |", "|---|---|---|---|"]
    for _, r in sp.iterrows():
        raw = "–" if pd.isna(r.B_raw) else f"{r.B_raw:.3f}"
        nor = "–" if pd.isna(r.B_normalized) else f"{r.B_normalized:.3f}"
        lines.append(f"| {r.dataset} | {r.pair} | {raw} | {nor} |{' ' + r.note if r.note else ''}")
    lines += ["", "## Pairwise same-class agreement across phantoms", "",
              f"raw mean {pwd.B_raw.mean():.3f}, normalized mean {pwd.B_normalized.mean():.3f}; "
              f"improved in {int((pwd.change > 0).sum())} of {len(pwd)} phantom pairs.", "",
              "Note: dividing every voxel of a phantom by one constant cannot change class "
              "separation *within* that phantom. Normalisation can only affect pooled or "
              "cross-phantom comparisons.", ""]
    (OUT / "summary.md").write_text("\n".join(lines))
    print(f"\nsaved {len(list(OUT.iterdir()))} files to {OUT}")


main()
