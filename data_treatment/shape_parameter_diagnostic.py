#!/usr/bin/env python3
"""Do any dimensionless shape parameters transfer between phantoms?

Read-only with respect to every existing file. Nothing is renormalised, no
classifier is trained, and no stored fit result is altered. The 42,685 Pill2
voxels fitted later stored only the AIC-selected parameters, so the identical
deterministic fit is re-run purely to recover both models' parameters; the
result is verified against what is stored before use, and written to a new file
so this recovery never has to happen again.

Candidates are limited to quantities that are dimensionless, and so carry no
phantom intensity scale by construction. ``rs * TE_max`` and
``exp(-rs * TE_max)`` are deliberately excluded: all four phantoms share the
same 26-point 8-208 ms echo grid, so both are monotone rescalings of rs and
would report rs's separation under a new name.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MRI = DATA_ROOT
FITS = MRI / "i0_rs" / "Segmentations"
P2_VOL = MRI / "i0_rs" / "Volume_for_testing" / "Pill2"
P2_SEG = MRI / "Normalized_data_i0_rs" / "Segmentations" / "Pill2"
P2_COMPLETE = MRI / "i0_rs" / "Volume_for_testing" / "Pill2_segmentation_complete" / "voxels_complete.csv.gz"
DICOM = MRI / "Volumes"
OUT = MRI / "Training" / "diagnosis" / "shape_parameters"
RECOVERED = OUT / "pill2_recovered_fit_fields.csv.gz"

CLASSES = ["0", "0.05", "0.1", "0.2", "0.25", "0.3", "tissue"]
SAMPLES = ["Syringes", "3D", "Pill1", "Pill2"]
FOCUS = [("0.2", "0.25"), ("tissue", "0.1")]
SEED, WORKERS, EPS = 42, 9, 1e-9

CANDIDATES = {
    "B_frac": "B / (I0_B + B)  — Model-B floor as a fraction of the TE=0 signal",
    "rs_ratio": "rs_B / rs_noB  — how much the rate changes when a floor is allowed",
    "I0_ratio": "I0_B / I0_noB  — how much the amplitude changes when a floor is allowed",
    "RMSE_rel": "RMSE_selected / I0_selected  — residual as a fraction of signal",
    "R2_selected": "R2 of the chosen model",
    "B_ratio": "B_selected / I0_selected  (reference, tested before)",
    "dAIC": "AIC_B - AIC_A  (reference, tested before)",
    "dR2": "R2_B - R2_noB  (reference, tested before)",
}
NEW = ["B_frac", "rs_ratio", "I0_ratio", "RMSE_rel", "R2_selected"]


def derive(d: pd.DataFrame) -> pd.DataFrame:
    """Build every candidate from raw two-model fit output."""
    a = d.model_selected.to_numpy() == 1
    I0_sel = np.where(a, d.I0_noB, d.I0_B)
    rmse_sel = np.where(a, d.RMSE_noB, d.RMSE_B)
    B_sel = np.where(a, 0.0, d.B)
    denom = lambda v: np.where(np.abs(v) < EPS, np.nan, v)
    return pd.DataFrame({
        "B_frac": d.B / denom(d.I0_B + d.B),
        "rs_ratio": d.rs_B / denom(d.rs_noB),
        "I0_ratio": d.I0_B / denom(d.I0_noB),
        "RMSE_rel": rmse_sel / denom(I0_sel),
        "R2_selected": np.where(a, d.R2_noB, d.R2_B),
        "B_ratio": B_sel / denom(I0_sel),
        "dAIC": d.AIC_B - d.AIC_noB,
        "dR2": d.R2_B - d.R2_noB,
    })


def recover_pill2(complete, series_factory, fit_signals):
    """Return raw fit fields for the 42,685 voxels that stored only selections."""
    if RECOVERED.is_file():
        print(f"  reusing verified recovery at {RECOVERED.name}")
        return pd.read_csv(RECOVERED)
    missing = complete[complete.origin == "newly_fitted"].reset_index(drop=True)
    series = series_factory()
    try:
        te = [e.value for e in series.echotimes]
        z = missing.z.to_numpy(np.intp); y = missing.y.to_numpy(np.intp); x = missing.x.to_numpy(np.intp)
        sig = np.zeros((len(missing), len(series)), np.float32)
        for c in range(len(series)):
            sig[:, c] = series.volume(c).data[z, y, x]
    finally:
        series.close()
    r = fit_signals(te, sig, workers=WORKERS)
    chose_a = r["model_selected"] == 1
    same = {
        "model_selected": bool(np.array_equal(r["model_selected"].astype(np.int8),
                                              missing.model_selected.to_numpy(np.int8))),
        "rs_selected": bool(np.allclose(np.where(chose_a, r["rs_noB"], r["rs_B"]),
                                        missing.rs_selected, rtol=1e-5, atol=1e-8)),
        "I0_selected": bool(np.allclose(np.where(chose_a, r["I0_noB"], r["I0_B"]),
                                        missing.I0_selected, rtol=1e-5, atol=1e-4)),
        "B_selected": bool(np.allclose(np.where(chose_a, 0.0, r["B"]),
                                       missing.B_selected, rtol=1e-5, atol=1e-4)),
    }
    print(f"  recovery reproduces the stored parameters: {same}")
    assert all(same.values()), "recovery changed the stored parameters"
    out = pd.DataFrame({k: r[k] for k in ("I0_noB", "rs_noB", "I0_B", "rs_B", "B",
                                          "R2_noB", "R2_B", "RMSE_noB", "RMSE_B",
                                          "AIC_noB", "AIC_B", "model_selected")})
    for c in ("z", "y", "x"):
        out[c] = missing[c].to_numpy()
    out["material"] = missing.material.astype(str).to_numpy()
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(RECOVERED, index=False, float_format="%.7g", compression="gzip")
    return out


def main() -> int:
    from echoviewer import EchoSeries
    from echoviewer.diagnostics import LABEL_COLORS, _figure, _save, iqr, overlap_1d
    from echoviewer.fitting import fit_signals

    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- training phantoms, straight from the stored fit CSVs -------------
    frames = []
    for s in ("Syringes", "3D", "Pill1"):
        for seg in sorted(p for p in (FITS / s).iterdir() if p.is_dir()):
            m = seg.name.split("_", 2)[2]
            if m not in CLASSES:
                continue
            d = pd.read_csv(seg / "voxel_fit_data.csv")
            d = d[d.fit_success > 0]
            f = derive(d)
            f["sample"] = s; f["material"] = m
            frames.append(f)
            del d
    df = pd.concat(frames, ignore_index=True); del frames

    # ---- Pill2, complete ---------------------------------------------------
    complete = pd.read_csv(P2_COMPLETE)
    complete["material"] = complete.material.astype(str)
    print(f"Pill2 complete segmentation: {len(complete):,} voxels")
    rec = recover_pill2(complete, lambda: EchoSeries.from_sample(DICOM, "Pill2", cache_size=2),
                        fit_signals)
    names = ("I0_A", "rs_A", "I0_B", "rs_B", "B_B", "R2_A", "R2_B",
             "RMSE_A", "RMSE_B", "AIC_A", "AIC_B", "model_selected")
    maps = {n: np.load(P2_VOL / f"{n}.npy") for n in names}
    ex = complete[complete.origin == "existing"]
    z = ex.z.to_numpy(np.intp); y = ex.y.to_numpy(np.intp); x = ex.x.to_numpy(np.intp)
    p2_existing = pd.DataFrame({
        "I0_noB": maps["I0_A"][z, y, x], "rs_noB": maps["rs_A"][z, y, x],
        "I0_B": maps["I0_B"][z, y, x], "rs_B": maps["rs_B"][z, y, x], "B": maps["B_B"][z, y, x],
        "R2_noB": maps["R2_A"][z, y, x], "R2_B": maps["R2_B"][z, y, x],
        "RMSE_noB": maps["RMSE_A"][z, y, x], "RMSE_B": maps["RMSE_B"][z, y, x],
        "AIC_noB": maps["AIC_A"][z, y, x], "AIC_B": maps["AIC_B"][z, y, x],
        "model_selected": maps["model_selected"][z, y, x]})
    p2_existing["material"] = ex.material.to_numpy()
    p2 = pd.concat([p2_existing, rec[list(p2_existing.columns)]], ignore_index=True)
    f2 = derive(p2)
    f2["sample"] = "Pill2"; f2["material"] = p2.material.astype(str).to_numpy()
    df = pd.concat([df, f2], ignore_index=True)
    df = df[df.material.isin(CLASSES)].reset_index(drop=True)
    print(f"  Pill2 candidates computed for {len(f2):,} of {len(complete):,} voxels "
          f"({100*len(f2)/len(complete):.1f}%)")
    print(f"training {int((df['sample']!='Pill2').sum()):,} voxels")

    present = {s: sorted(set(df[df["sample"] == s].material), key=CLASSES.index) for s in SAMPLES}
    for s in SAMPLES:
        print(f"  {s:9s} {present[s]}")

    def values(sample, material, feat):
        v = df[(df["sample"] == sample) & (df.material == material)][feat].to_numpy(np.float64)
        return v[np.isfinite(v)]

    # ---- 1. distribution stats --------------------------------------------
    stats = []
    for s in SAMPLES:
        for m in present[s]:
            for feat in CANDIDATES:
                v = values(s, m, feat)
                if len(v) < 10:
                    continue
                q1, q3 = iqr(v)
                stats.append({"sample": s, "material": m, "candidate": feat, "n": len(v),
                              "median": float(np.median(v)), "q1": q1, "q3": q3,
                              "p05": float(np.percentile(v, 5)),
                              "p95": float(np.percentile(v, 95))})
    pd.DataFrame(stats).round(5).to_csv(OUT / "stats_by_class_phantom.csv", index=False)

    # ---- 2. phantom invariance (same class, different phantom) ------------
    inv_rows = []
    for feat in CANDIDATES:
        for m in CLASSES:
            hosts = [s for s in SAMPLES if m in present[s]]
            for i in range(len(hosts)):
                for j in range(i + 1, len(hosts)):
                    a, b = values(hosts[i], m, feat), values(hosts[j], m, feat)
                    if len(a) < 10 or len(b) < 10:
                        continue
                    inv_rows.append({"candidate": feat, "material": m,
                                     "sample_a": hosts[i], "sample_b": hosts[j],
                                     "overlap": overlap_1d(a, b)})
    inv = pd.DataFrame(inv_rows)
    inv.round(4).to_csv(OUT / "phantom_invariance_pairs.csv", index=False)
    inv_mean = inv.groupby("candidate").overlap.mean()

    # ---- 3. separation, within phantom and across phantoms ----------------
    sep_rows = []
    for feat in CANDIDATES:
        for ca, cb in FOCUS:
            for sa in SAMPLES:
                if ca not in present[sa]:
                    continue
                for sb in SAMPLES:
                    if cb not in present[sb]:
                        continue
                    a, b = values(sa, ca, feat), values(sb, cb, feat)
                    if len(a) < 10 or len(b) < 10:
                        continue
                    sep_rows.append({"candidate": feat, "pair": f"{ca} vs {cb}",
                                     "sample_a": sa, "sample_b": sb,
                                     "kind": "within-phantom" if sa == sb else "cross-phantom",
                                     "overlap": overlap_1d(a, b)})
            pa = df[(df["sample"] != "Pill2") & (df.material == ca)][feat].to_numpy(np.float64)
            pb = df[(df["sample"] != "Pill2") & (df.material == cb)][feat].to_numpy(np.float64)
            pa, pb = pa[np.isfinite(pa)], pb[np.isfinite(pb)]
            if len(pa) >= 10 and len(pb) >= 10:
                sep_rows.append({"candidate": feat, "pair": f"{ca} vs {cb}",
                                 "sample_a": "training pooled", "sample_b": "training pooled",
                                 "kind": "pooled training", "overlap": overlap_1d(pa, pb)})
    sep = pd.DataFrame(sep_rows)
    sep.round(4).to_csv(OUT / "separation_within_and_across_phantoms.csv", index=False)

    summary = []
    for feat in CANDIDATES:
        row = {"candidate": feat, "new": feat in NEW,
               "phantom_invariance_mean_overlap": float(inv_mean.get(feat, np.nan))}
        for ca, cb in FOCUS:
            s = sep[(sep.candidate == feat) & (sep.pair == f"{ca} vs {cb}")]
            tag = f"{ca}_vs_{cb}".replace(".", "")
            for kind, key in (("pooled training", "pooled"), ("within-phantom", "within"),
                              ("cross-phantom", "cross")):
                sub = s[s.kind == kind].overlap
                row[f"{tag}_{key}"] = float(sub.mean()) if len(sub) else np.nan
            row[f"{tag}_cross_worst"] = (float(s[s.kind == "cross-phantom"].overlap.max())
                                         if (s.kind == "cross-phantom").any() else np.nan)
        summary.append(row)
    sm = pd.DataFrame(summary).sort_values("phantom_invariance_mean_overlap", ascending=False)
    sm.round(4).to_csv(OUT / "candidate_summary.csv", index=False)
    print("\nCANDIDATE SUMMARY (overlap: lower = better separated, higher = more invariant)")
    print(sm.round(3).to_string(index=False))

    # ---- figures ----------------------------------------------------------
    for feat, desc in CANDIDATES.items():
        fig = _figure(figsize=(18, 5.2))
        axes = fig.subplots(1, 4)
        allv = df[feat].to_numpy(np.float64); allv = allv[np.isfinite(allv)]
        lo, hi = np.percentile(allv, [1, 99])
        if hi - lo < 1e-12:
            lo, hi = lo - 1, hi + 1
        for ax, s in zip(axes, SAMPLES):
            data = [np.clip(values(s, m, feat), lo, hi) for m in present[s]]
            keep = [(m, v) for m, v in zip(present[s], data) if len(v) >= 10]
            if not keep:
                ax.set_axis_off(); continue
            labels, data = zip(*keep)
            parts = ax.violinplot(list(data), showextrema=False, widths=.85)
            for body, m in zip(parts["bodies"], labels):
                body.set_facecolor(LABEL_COLORS[m]); body.set_alpha(.7)
            for pos, v in enumerate(data, start=1):
                q1, q3 = iqr(v)
                ax.vlines(pos, q1, q3, color="black", lw=3)
                ax.plot(pos, np.median(v), "o", color="white", markeredgecolor="black", ms=5)
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, rotation=30, fontsize=8)
            ax.set_ylim(lo, hi); ax.grid(alpha=.25, axis="y")
            ax.set_title(f"{s} (n={int((df['sample']==s).sum()):,})", fontsize=10)
            if s == SAMPLES[0]:
                ax.set_ylabel(feat)
        fig.suptitle(f"{feat} — {desc}   (shared y axis, clipped to 1–99th percentile)",
                     fontsize=12)
        _save(fig, OUT / f"candidate_{feat}.png", [])

    order = list(sm.candidate)
    fig = _figure(figsize=(15, 6))
    axes = fig.subplots(1, 2)
    for ax, (ca, cb) in zip(axes, FOCUS):
        tag = f"{ca}_vs_{cb}".replace(".", "")
        idx = np.arange(len(order))
        for off, key, colour, lbl in ((-0.26, "pooled", "#9bb0e8", "pooled training"),
                                      (0.0, "within", "#4a6fe3", "within phantom"),
                                      (0.26, "cross", "#d94a4a", "cross phantom")):
            ax.bar(idx + off, sm[f"{tag}_{key}"].to_numpy(), width=.25, color=colour, label=lbl)
        ax.set_xticks(idx); ax.set_xticklabels(order, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("overlap (lower = better separated)")
        ax.set_ylim(0, 1.05); ax.grid(alpha=.25, axis="y"); ax.legend(fontsize=8)
        ax.set_title(f"{ca} vs {cb}")
    fig.suptitle("Separation holds up across phantoms only where the red bar matches the blue",
                 fontsize=13)
    _save(fig, OUT / "separation_within_vs_cross_phantom.png", [])

    fig = _figure(figsize=(11, 6))
    ax = fig.subplots()
    ax.bar(range(len(order)), sm.phantom_invariance_mean_overlap.to_numpy(),
           color=["#4a6fe3" if n else "#9aa4b8" for n in sm.new])
    for i, v in enumerate(sm.phantom_invariance_mean_overlap.to_numpy()):
        ax.text(i, v + .01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("mean same-class overlap across phantoms\n(higher = more phantom-invariant)")
    ax.set_ylim(0, 1); ax.grid(alpha=.25, axis="y")
    ax.set_title("Phantom invariance (blue = new candidate, grey = previously tested)")
    _save(fig, OUT / "phantom_invariance_summary.png", [])

    # ---- written summary ---------------------------------------------------
    lines = ["# Dimensionless shape parameters: do any transfer between phantoms?", "",
             "Read-only. Nothing renormalised, no classifier trained, no stored fit result "
             "altered. Overlap is the 1-D histogram overlap on complete voxel populations: "
             "0 = disjoint, 1 = identical.", "",
             f"Training {int((df['sample']!='Pill2').sum()):,} voxels; "
             f"Pill2 {len(f2):,} of {len(complete):,} complete segmented voxels. The 42,685 "
             "voxels that stored only AIC-selected parameters were recovered by re-running the "
             "identical deterministic fit, verified to reproduce every stored parameter, and "
             f"written to `{RECOVERED.name}`.", "",
             "`rs * TE_max` and `exp(-rs * TE_max)` are excluded: all four phantoms share the "
             "same 26-point 8–208 ms echo grid, so both are monotone rescalings of rs.", "",
             "## Class availability limits what can be tested", "",
             "| phantom | classes |", "|---|---|"]
    for s in SAMPLES:
        lines.append(f"| {s} | {', '.join(present[s])} |")
    lines += ["", "`0.2` and `0.25` coexist only in Syringes and 3D, so within-phantom "
              "separation is measurable in those two alone; Pill1 has 0.2 without 0.25 and "
              "Pill2 has 0.25 without 0.2. `tissue` exists only in Pill1 and Pill2, and "
              "coexists with `0.1` only in Pill2.", "",
              "## Candidates", "",
              "| candidate | definition |", "|---|---|"]
    for feat, desc in CANDIDATES.items():
        lines.append(f"| `{feat}`{' *(new)*' if feat in NEW else ''} | {desc} |")
    lines += ["", "## Results", "",
              "| candidate | phantom invariance | 0.2/0.25 pooled | within | cross | "
              "tissue/0.1 pooled | within | cross |", "|---|---|---|---|---|---|---|---|"]
    for _, r in sm.iterrows():
        f = lambda v: "–" if pd.isna(v) else f"{v:.3f}"
        lines.append(f"| `{r.candidate}` | {f(r.phantom_invariance_mean_overlap)} | "
                     f"{f(r['02_vs_025_pooled'])} | {f(r['02_vs_025_within'])} | "
                     f"{f(r['02_vs_025_cross'])} | {f(r['tissue_vs_01_pooled'])} | "
                     f"{f(r['tissue_vs_01_within'])} | {f(r['tissue_vs_01_cross'])} |")
    (OUT / "summary.md").write_text("\n".join(lines))

    (OUT / "summary.json").write_text(json.dumps({
        "candidates": CANDIDATES, "new_candidates": NEW,
        "excluded": {"rs*TE_max": "identical echo grid makes it a rescaling of rs",
                     "exp(-rs*TE_max)": "monotone transform of rs"},
        "n_training": int((df["sample"] != "Pill2").sum()),
        "n_pill2": int(len(f2)), "n_pill2_complete": int(len(complete)),
        "class_availability": present,
        "results": sm.to_dict(orient="records"),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2, default=float))
    print(f"\nsaved {len(list(OUT.iterdir()))} files to {OUT}")
    print(f"elapsed {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
