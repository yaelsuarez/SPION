"""READ-ONLY: do dAIC, selected RMSE, dR2 and B_ratio transfer across phantoms?

Everything is arithmetic on already-stored fit output. Nothing is refitted,
renormalised, trained or modified.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT  # noqa: E402
import numpy as np, pandas as pd

MRI = DATA_ROOT
FITS = MRI / "i0_rs" / "Segmentations"
P2_VOL = MRI / "i0_rs" / "Volume_for_testing" / "Pill2"
P2_SEG = MRI / "Normalized_data_i0_rs" / "Segmentations" / "Pill2"
P2_COMPLETE = MRI / "i0_rs" / "Volume_for_testing" / "Pill2_segmentation_complete" / "voxels_complete.csv.gz"
OUT = MRI / "Training" / "diagnosis" / "derived_features"

CLASSES = ["0", "0.05", "0.1", "0.2", "0.25", "0.3", "tissue"]
SAMPLES = ["Syringes", "3D", "Pill1", "Pill2"]
FEATURES = {"dAIC": "AIC_B - AIC_A  (<0 favours B)",
            "RMSE_selected": "RMSE of the chosen model",
            "dR2": "R2_B - R2_noB",
            "B_ratio": "B_selected / I0_selected"}
PAIRS = [("0.2", "0.25"), ("tissue", "0.1"), ("tissue", "0.05"),
         ("0.1", "0.2"), ("0.25", "0.3"), ("0", "0.1")]
SEED = 42


def derive(d):
    a = d.model_selected.to_numpy() == 1
    out = pd.DataFrame({
        "dAIC": d.AIC_B.to_numpy() - d.AIC_noB.to_numpy(),
        "RMSE_selected": np.where(a, d.RMSE_noB, d.RMSE_B),
        "dR2": d.R2_B.to_numpy() - d.R2_noB.to_numpy(),
        "B_ratio": np.where(a, 0.0, d.B.to_numpy() /
                            np.maximum(np.where(a, d.I0_noB, d.I0_B), 1e-9)),
    })
    return out


def main():
    import echoviewer  # noqa: F401
    from echoviewer.diagnostics import (LABEL_COLORS, _figure, _save, overlap_1d, iqr)

    OUT.mkdir(parents=True, exist_ok=True)

    # ---- training, straight from the fit CSVs -----------------------------
    rows = []
    for s in ("Syringes", "3D", "Pill1"):
        for seg in sorted(p for p in (FITS / s).iterdir() if p.is_dir()):
            m = seg.name.split("_", 2)[2]
            if m not in CLASSES:
                continue
            d = pd.read_csv(seg / "voxel_fit_data.csv")
            d = d[d.fit_success > 0]
            f = derive(d)
            f["sample"] = s; f["material"] = m
            rows.append(f)
            del d
    df = pd.concat(rows, ignore_index=True); del rows

    # ---- Pill2: only where AIC/RMSE/R2 exist (the original volume fit) -----
    maps = {n: np.load(P2_VOL / f"{n}.npy") for n in
            ("AIC_A", "AIC_B", "R2_A", "R2_B", "RMSE_A", "RMSE_B",
             "model_selected", "I0_A", "I0_B", "B_B", "fit_success")}
    lab = pd.concat([pd.read_csv(p / "voxels.csv.gz", usecols=["x", "y", "z"]).assign(
        material=str(p.name.split("_", 2)[2]))
        for p in sorted(q for q in P2_SEG.iterdir() if q.is_dir())], ignore_index=True)
    z, y, x = lab.z.to_numpy(np.intp), lab.y.to_numpy(np.intp), lab.x.to_numpy(np.intp)
    ok = maps["fit_success"][z, y, x] > 0
    z, y, x, lab = z[ok], y[ok], x[ok], lab[ok]
    p2 = pd.DataFrame({"AIC_noB": maps["AIC_A"][z, y, x], "AIC_B": maps["AIC_B"][z, y, x],
                       "R2_noB": maps["R2_A"][z, y, x], "R2_B": maps["R2_B"][z, y, x],
                       "RMSE_noB": maps["RMSE_A"][z, y, x], "RMSE_B": maps["RMSE_B"][z, y, x],
                       "model_selected": maps["model_selected"][z, y, x],
                       "I0_noB": maps["I0_A"][z, y, x], "I0_B": maps["I0_B"][z, y, x],
                       "B": maps["B_B"][z, y, x]})
    f2 = derive(p2); f2["sample"] = "Pill2"; f2["material"] = lab.material.to_numpy()
    df = pd.concat([df, f2], ignore_index=True)
    complete = pd.read_csv(P2_COMPLETE, usecols=["material", "B_selected", "I0_selected"])
    print(f"training {len(df[df['sample']!='Pill2']):,} voxels")
    print(f"Pill2 with AIC/RMSE/R2: {len(f2):,} of 384,315 segmented "
          f"({100*len(f2)/384315:.1f}%) — the 42,685 refitted voxels lack those columns")
    print("  Pill2 coverage by class:")
    cov = f2.material.value_counts()
    full = complete.material.astype(str).value_counts()
    for c in CLASSES:
        if c in full.index:
            print(f"    {c:7s} {int(cov.get(c,0)):>7,} of {int(full[c]):>7,} "
                  f"({100*cov.get(c,0)/full[c]:5.1f}%)")

    # ---- 1. per class x phantom stats -------------------------------------
    stats = []
    for (s, m), g in df.groupby(["sample", "material"]):
        for feat in FEATURES:
            v = g[feat].to_numpy(np.float64)
            v = v[np.isfinite(v)]
            if not len(v):
                continue
            q1, q3 = iqr(v)
            stats.append({"sample": s, "material": m, "feature": feat, "n": len(v),
                          "median": float(np.median(v)), "q1": q1, "q3": q3, "iqr": q3 - q1,
                          "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95))})
    st = pd.DataFrame(stats)
    st.round(5).to_csv(OUT / "feature_stats_by_class_phantom.csv", index=False)

    # ---- 2. phantom-to-phantom shift, same class --------------------------
    shift = []
    for feat in FEATURES:
        for m in CLASSES:
            present = [s for s in SAMPLES if len(df[(df["sample"] == s) & (df.material == m)])]
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    a = df[(df["sample"] == present[i]) & (df.material == m)][feat].to_numpy(np.float64)
                    b = df[(df["sample"] == present[j]) & (df.material == m)][feat].to_numpy(np.float64)
                    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
                    if len(a) < 10 or len(b) < 10:
                        continue
                    shift.append({"feature": feat, "material": m,
                                  "sample_a": present[i], "sample_b": present[j],
                                  "overlap": overlap_1d(a, b),
                                  "median_a": float(np.median(a)), "median_b": float(np.median(b))})
    sh = pd.DataFrame(shift)
    sh.round(4).to_csv(OUT / "same_class_cross_phantom.csv", index=False)
    inv = sh.groupby("feature").overlap.agg(["mean", "median", "count"]).round(3)
    print("\n2. PHANTOM INVARIANCE (same class, different phantom; higher = agrees)")
    print(inv.to_string())

    # ---- 3. class separation ----------------------------------------------
    sep = []
    for feat in FEATURES:
        for label, sub in (("training pooled", df[df["sample"] != "Pill2"]),
                           ("Pill2", df[df["sample"] == "Pill2"])):
            for a, b in PAIRS:
                va = sub[sub.material == a][feat].to_numpy(np.float64)
                vb = sub[sub.material == b][feat].to_numpy(np.float64)
                va, vb = va[np.isfinite(va)], vb[np.isfinite(vb)]
                sep.append({"feature": feat, "dataset": label, "pair": f"{a} vs {b}",
                            "overlap": overlap_1d(va, vb) if len(va) > 10 and len(vb) > 10 else np.nan})
    sp = pd.DataFrame(sep)
    sp.round(4).to_csv(OUT / "class_pair_separation.csv", index=False)
    print("\n3. CLASS SEPARATION (lower = better separated)")
    piv = sp.pivot_table(index=["feature", "pair"], columns="dataset", values="overlap")
    print(piv.round(3).to_string())

    # ---- figures -----------------------------------------------------------
    for feat, desc in FEATURES.items():
        fig = _figure(figsize=(18, 5.2))
        axes = fig.subplots(1, 4)
        v_all = df[feat].to_numpy(np.float64); v_all = v_all[np.isfinite(v_all)]
        lo, hi = np.percentile(v_all, [1, 99])
        for ax, s in zip(axes, SAMPLES):
            sub = df[df["sample"] == s]
            present = [c for c in CLASSES if (sub.material == c).any()]
            data = [np.clip(sub[sub.material == c][feat].to_numpy(np.float64), lo, hi)
                    for c in present]
            parts = ax.violinplot(data, showextrema=False, widths=.85)
            for body, c in zip(parts["bodies"], present):
                body.set_facecolor(LABEL_COLORS[c]); body.set_alpha(.7)
            for pos, v in enumerate(data, start=1):
                q1, q3 = iqr(v)
                ax.vlines(pos, q1, q3, color="black", lw=3)
                ax.plot(pos, np.median(v), "o", color="white", markeredgecolor="black", ms=5)
            ax.set_xticks(range(1, len(present) + 1)); ax.set_xticklabels(present, rotation=30)
            ax.set_ylim(lo, hi); ax.grid(alpha=.25, axis="y")
            ax.set_title(f"{s} (n={len(sub):,})", fontsize=10)
            if s == SAMPLES[0]:
                ax.set_ylabel(feat)
        fig.suptitle(f"{feat} — {desc}   (identical y axis, clipped to 1–99th percentile)",
                     fontsize=13)
        _save(fig, OUT / f"{feat}_by_class_and_phantom.png", [])

    fig = _figure(figsize=(11, 5.5))
    ax = fig.subplots()
    order = list(FEATURES)
    m = inv.reindex(order)["mean"]
    ax.bar(range(len(order)), m.values, color="#4a6fe3")
    for i, v in enumerate(m.values):
        ax.text(i, v + .01, f"{v:.3f}", ha="center")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=20)
    ax.set_ylabel("mean same-class overlap across phantoms\n(higher = more phantom-invariant)")
    ax.set_ylim(0, 1); ax.grid(alpha=.25, axis="y")
    ax.set_title("Phantom invariance of the derived features")
    _save(fig, OUT / "phantom_invariance_summary.png", [])

    lines = ["# Derived-feature transfer diagnostic", "",
             "Read-only; arithmetic on stored fit output only. Nothing refitted, "
             "renormalised, trained or modified.", "",
             f"Training {len(df[df['sample']!='Pill2']):,} voxels. "
             f"**Pill2 covers {len(f2):,} of 384,315 segmented voxels ({100*len(f2)/384315:.1f}%)** "
             "— the 42,685 voxels refitted later have model_selected/I0/rs/B stored but not "
             "AIC, RMSE or R2, so dAIC, RMSE_selected and dR2 cannot be computed for them "
             "without refitting.", "",
             "## Phantom invariance (same class, different phantom; higher = agrees)", "",
             "| feature | mean overlap | median | n pairs |", "|---|---|---|---|"]
    for feat in order:
        r = inv.loc[feat]
        lines.append(f"| {feat} | {r['mean']:.3f} | {r['median']:.3f} | {int(r['count'])} |")
    lines += ["", "## Class separation (lower = better)", "",
              "| feature | pair | training | Pill2 |", "|---|---|---|---|"]
    for feat in order:
        for a, b in PAIRS:
            row = sp[(sp.feature == feat) & (sp.pair == f"{a} vs {b}")]
            tr = row[row.dataset == "training pooled"].overlap
            p2v = row[row.dataset == "Pill2"].overlap
            fmt = lambda s: "–" if not len(s) or pd.isna(s.iloc[0]) else f"{s.iloc[0]:.3f}"
            lines.append(f"| {feat} | {a} vs {b} | {fmt(tr)} | {fmt(p2v)} |")
    (OUT / "summary.md").write_text("\n".join(lines))
    print(f"\nsaved {len(list(OUT.iterdir()))} files to {OUT}")


main()
