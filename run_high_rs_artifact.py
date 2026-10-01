#!/usr/bin/env python3
"""Diagnose the high-rs branch that Model B produces at high SPION concentration.

Purely diagnostic: reads the existing fits, writes PNGs and a summary under
``i0_rs/model_diagnostics/high_rs_artifact``. The only fitting done here is a
handful of refits of *representative voxels* to test candidate bounds; no
existing output is touched.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_ROOT  # noqa: E402

import numpy as np  # noqa: E402

MRI = DATA_ROOT
FIT_ROOT = MRI / "i0_rs" / "Segmentations"
SEG_ROOT = MRI / "Segmentations"
OUT = MRI / "i0_rs" / "model_diagnostics" / "high_rs_artifact"

#: Working cut for the branch. Section 7 checks whether the data supports it.
BRANCH_RS = 0.15
SEED = 42
MAX_PLOT = 120_000


def main() -> int:
    import echoviewer  # noqa: F401
    import pandas as pd
    from echoviewer.diagnostics import (LABEL_COLORS, LABEL_ORDER, _figure, _save,
                                        iter_fit_dirs, load_voxel_curves, read_voxels,
                                        subsample, iqr)
    from echoviewer.fitting import model_no_baseline, model_with_baseline

    started = time.time()
    for name in ("I0_vs_rs", "decay_curves", "residuals", "spatial_maps",
                 "signal_vs_TE", "constraint_analysis", "summary"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    rng = np.random.default_rng(SEED)

    NEEDED = ("x", "y", "z", "I0_noB", "rs_noB", "I0_B", "rs_B", "B",
              "AIC_noB", "AIC_B", "R2_noB", "R2_B", "model_selected", "fit_success")

    # ---- collect -------------------------------------------------------
    print("Reading fits")
    pools: dict[str, dict[str, list]] = {}
    frames_03: dict[tuple[str, str], pd.DataFrame] = {}
    rs_by_sample: dict[str, list] = {}

    for sample, seg_name, label, seg_dir in iter_fit_dirs(FIT_ROOT):
        frame = read_voxels(seg_dir, NEEDED)
        chose_a = frame["model_selected"].to_numpy() == 1
        rs_sel = np.where(chose_a, frame["rs_noB"], frame["rs_B"]).astype(np.float32)
        i0_sel = np.where(chose_a, frame["I0_noB"], frame["I0_B"]).astype(np.float32)
        b_sel = np.where(chose_a, 0.0, frame["B"]).astype(np.float32)

        store = pools.setdefault(label, {"rs": [], "I0": [], "B": []})
        store["rs"].append(rs_sel)
        store["I0"].append(i0_sel)
        store["B"].append(b_sel)
        rs_by_sample.setdefault(sample, []).append(rs_sel)

        if label == "0.3":
            frame = frame.assign(rs_sel=rs_sel, I0_sel=i0_sel, B_sel=b_sel)
            frames_03[(sample, seg_name)] = frame
        else:
            del frame
    pools = {k: {n: np.concatenate(v) for n, v in s.items()} for k, s in pools.items()}
    labels = [l for l in LABEL_ORDER if l in pools]
    rs_by_sample = {k: np.concatenate(v) for k, v in rs_by_sample.items()}

    # ---- 1. I0 vs rs ---------------------------------------------------
    print("1. I0 vs rs")
    rs_lim = (0, float(np.percentile(np.concatenate([pools[l]["rs"] for l in labels]), 99.9)) * 1.05)
    i0_lim = (0, float(np.percentile(np.concatenate([pools[l]["I0"] for l in labels]), 99.9)) * 1.05)

    figure = _figure(figsize=(10, 8.5))
    axes = figure.subplots()
    total = drawn = 0
    for label in labels:
        x, y = subsample([pools[label]["rs"], pools[label]["I0"]], MAX_PLOT,
                         np.random.default_rng(SEED))
        total += pools[label]["rs"].size
        drawn += x.size
        axes.scatter(x, y, s=2, alpha=0.12, linewidths=0, color=LABEL_COLORS[label],
                     label=f"{label} (n={pools[label]['rs'].size:,})")
    axes.set(xlabel="rs_selected (1/ms)", ylabel="I0_selected (a.u.)", xlim=rs_lim, ylim=i0_lim)
    axes.axvline(BRANCH_RS, color="black", ls="--", lw=1.2, label=f"rs = {BRANCH_RS}")
    axes.set_title(f"Model-selected I0 vs rs, all materials\n{drawn:,} of {total:,} plotted, seed {SEED}")
    legend = axes.legend(markerscale=6, fontsize=8)
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
    axes.grid(alpha=0.25)
    _save(figure, OUT / "I0_vs_rs" / "all_materials.png", saved)

    for label in labels:
        figure = _figure(figsize=(8.5, 7))
        axes = figure.subplots()
        x, y = subsample([pools[label]["rs"], pools[label]["I0"]], MAX_PLOT,
                         np.random.default_rng(SEED))
        axes.scatter(x, y, s=3, alpha=0.15, linewidths=0, color=LABEL_COLORS[label])
        axes.axvline(BRANCH_RS, color="black", ls="--", lw=1.2)
        above = int((pools[label]["rs"] > BRANCH_RS).sum())
        axes.set(xlabel="rs_selected (1/ms)", ylabel="I0_selected (a.u.)",
                 xlim=rs_lim, ylim=i0_lim)
        axes.set_title(f"{label} — {pools[label]['rs'].size:,} voxels, "
                       f"{above:,} above rs={BRANCH_RS} ({100*above/pools[label]['rs'].size:.2f}%)\n"
                       f"{x.size:,} plotted, seed {SEED}")
        axes.grid(alpha=0.25)
        _save(figure, OUT / "I0_vs_rs" / f"material_{label.replace('.', 'p')}.png", saved)

    figure = _figure(figsize=(10, 8))
    axes = figure.subplots()
    colours = {"3D": "#4a6fe3", "Syringes": "#c9314e", "Pill2": "#e8b647"}
    for (sample, seg_name), frame in sorted(frames_03.items()):
        x, y = subsample([frame["rs_sel"].to_numpy(), frame["I0_sel"].to_numpy()],
                         MAX_PLOT, np.random.default_rng(SEED))
        axes.scatter(x, y, s=4, alpha=0.25, linewidths=0, color=colours.get(sample, "#666"),
                     label=f"{sample} (n={len(frame):,})")
    axes.axvline(BRANCH_RS, color="black", ls="--", lw=1.2, label=f"rs = {BRANCH_RS}")
    axes.set(xlabel="rs_selected (1/ms)", ylabel="I0_selected (a.u.)")
    axes.set_title("0.3 mg/mL across phantoms — the branch is not present everywhere")
    legend = axes.legend(markerscale=5)
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
    axes.grid(alpha=0.25)
    _save(figure, OUT / "I0_vs_rs" / "concentration_0p3_by_sample.png", saved)

    # ---- 2. branch statistics -----------------------------------------
    print("2. branch statistics")
    rows = []
    for (sample, seg_name), frame in sorted(frames_03.items()):
        branch = frame["rs_sel"].to_numpy() > BRANCH_RS
        for tag, mask in (("normal", ~branch), ("high_rs", branch)):
            if mask.sum() == 0:
                continue
            part = frame[mask]
            entry = {"sample": sample, "segmentation": seg_name, "group": tag,
                     "n": int(mask.sum()),
                     "pct_of_segmentation": 100 * float(mask.mean())}
            for field, key in (("rs_sel", "rs"), ("I0_sel", "I0"), ("B_sel", "B")):
                q1, q3 = iqr(part[field].to_numpy())
                entry[f"median_{key}"] = float(np.median(part[field]))
                entry[f"q1_{key}"] = q1
                entry[f"q3_{key}"] = q3
            entry["median_R2_B"] = float(np.median(part["R2_B"]))
            entry["median_R2_A"] = float(np.median(part["R2_noB"]))
            rows.append(entry)
    branch_table = pd.DataFrame(rows)

    figure = _figure(figsize=(10, 8))
    axes = figure.subplots()
    for (sample, seg_name), frame in sorted(frames_03.items()):
        branch = frame["rs_sel"].to_numpy() > BRANCH_RS
        for mask, colour, tag in ((~branch, "#4a6fe3", "normal"), (branch, "#c9314e", "high-rs")):
            if mask.sum() == 0:
                continue
            x, y = subsample([frame["rs_sel"].to_numpy()[mask], frame["I0_sel"].to_numpy()[mask]],
                             MAX_PLOT, np.random.default_rng(SEED))
            axes.scatter(x, y, s=5, alpha=0.3, linewidths=0, color=colour,
                         label=f"{tag}" if (sample, tag) == ("3D", tag) else None)
    axes.axvline(BRANCH_RS, color="black", ls="--", lw=1.2)
    axes.set(xlabel="rs_selected (1/ms)", ylabel="I0_selected (a.u.)")
    axes.set_title("0.3 mg/mL: normal population (blue) vs high-rs branch (red)")
    axes.grid(alpha=0.25)
    _save(figure, OUT / "I0_vs_rs" / "concentration_0p3_branch_highlighted.png", saved)

    # ---- 3/4/6. curves, residuals, signal vs TE -----------------------
    print("3-6. decay curves, residuals, signal vs TE")
    curve_store = {}
    for (sample, seg_name), frame in sorted(frames_03.items()):
        branch = frame["rs_sel"].to_numpy() > BRANCH_RS
        if branch.sum() < 20:
            continue
        picks = {}
        for tag, mask in (("normal", ~branch), ("high_rs", branch)):
            index = np.flatnonzero(mask)
            chosen = np.random.default_rng(SEED).choice(index, size=min(60, index.size), replace=False)
            picks[tag] = chosen
        coords = [(int(frame.iloc[i]["z"]), int(frame.iloc[i]["y"]), int(frame.iloc[i]["x"]))
                  for tag in picks for i in picks[tag]]
        te, curves = load_voxel_curves(SEG_ROOT / sample / "volumes" / seg_name, coords)
        split = len(picks["normal"])
        curve_store[(sample, seg_name)] = {
            "te": te, "normal": curves[:split], "high_rs": curves[split:],
            "rows_normal": frame.iloc[picks["normal"]].reset_index(drop=True),
            "rows_high": frame.iloc[picks["high_rs"]].reset_index(drop=True),
        }

    for (sample, seg_name), store in curve_store.items():
        te = store["te"]
        for tag, key in (("normal", "rows_normal"), ("high_rs", "rows_high")):
            figure = _figure(figsize=(15, 8))
            panels = figure.subplots(2, 3).ravel()
            for axes, position in zip(panels, range(6)):
                row = store[key].iloc[position]
                curve = store[tag][position].astype(float)
                axes.plot(te, curve, "ko", markersize=4, label="measured")
                axes.plot(te, model_no_baseline(te, row["I0_noB"], row["rs_noB"]), "-",
                          color="#4a6fe3", lw=1.5,
                          label=f"A: I0={row['I0_noB']:.0f} rs={row['rs_noB']:.4f}\n"
                                f"   R²={row['R2_noB']:.4f} AIC={row['AIC_noB']:.1f}")
                axes.plot(te, model_with_baseline(te, row["I0_B"], row["rs_B"], row["B"]), "-",
                          color="#c9314e", lw=1.5,
                          label=f"B: I0={row['I0_B']:.0f} rs={row['rs_B']:.4f} B={row['B']:.1f}\n"
                                f"   R²={row['R2_B']:.4f} AIC={row['AIC_B']:.1f}")
                chosen = "A" if row["model_selected"] == 1 else "B"
                axes.set_title(f"(x,y,z)=({int(row['x'])},{int(row['y'])},{int(row['z'])})  "
                               f"selected {chosen}  ΔAIC={row['AIC_B']-row['AIC_noB']:+.1f}",
                               fontsize=8)
                axes.set(xlabel="TE (ms)", ylabel="intensity")
                axes.legend(fontsize=6)
                axes.grid(alpha=0.25)
            figure.suptitle(f"{sample} / {seg_name} — {tag} 0.3 mg/mL voxels", fontsize=12)
            _save(figure, OUT / "decay_curves" / f"{sample}_{tag}.png", saved)

            figure = _figure(figsize=(15, 5))
            panels = figure.subplots(1, 3)
            for axes, position in zip(panels, range(3)):
                row = store[key].iloc[position]
                curve = store[tag][position].astype(float)
                for model, fitted, colour in (
                    ("A", model_no_baseline(te, row["I0_noB"], row["rs_noB"]), "#4a6fe3"),
                    ("B", model_with_baseline(te, row["I0_B"], row["rs_B"], row["B"]), "#c9314e"),
                ):
                    axes.plot(te, curve - fitted, "o-", markersize=3, lw=1, color=colour,
                              label=f"Model {model} (mean {np.mean(curve-fitted):+.1f})")
                axes.axhline(0, color="black", lw=1)
                axes.set(xlabel="TE (ms)", ylabel="measured − fitted")
                axes.set_title(f"(x,y,z)=({int(row['x'])},{int(row['y'])},{int(row['z'])})", fontsize=9)
                axes.legend(fontsize=7)
                axes.grid(alpha=0.25)
            figure.suptitle(f"Residuals — {sample} / {seg_name}, {tag}", fontsize=12)
            _save(figure, OUT / "residuals" / f"{sample}_{tag}.png", saved)

        figure = _figure(figsize=(10, 6))
        axes = figure.subplots()
        for tag, colour in (("normal", "#4a6fe3"), ("high_rs", "#c9314e")):
            block = store[tag].astype(float)
            median = np.median(block, axis=0)
            low = np.percentile(block, 10, axis=0)
            high = np.percentile(block, 90, axis=0)
            axes.plot(te, median, "o-", color=colour, label=f"{tag} (median, n={block.shape[0]})")
            axes.fill_between(te, low, high, color=colour, alpha=0.2, label=f"{tag} 10–90%")
        axes.set(xlabel="Echo time (ms)", ylabel="intensity (a.u.)", yscale="log")
        axes.set_title(f"{sample} / {seg_name} — how fast each population reaches the noise floor")
        axes.legend(fontsize=8)
        axes.grid(alpha=0.25, which="both")
        _save(figure, OUT / "signal_vs_TE" / f"{sample}_signal_vs_TE.png", saved)

    # ---- 5. spatial ----------------------------------------------------
    print("5. spatial maps")
    for (sample, seg_name), frame in sorted(frames_03.items()):
        branch = frame["rs_sel"].to_numpy() > BRANCH_RS
        if branch.sum() < 20:
            continue
        first_echo = sorted((SEG_ROOT / sample / "volumes" / seg_name).glob("*/*.npz"),
                            key=lambda p: float(p.parent.name))[0]
        with np.load(first_echo) as archive:
            reference = archive["volume"] if "volume" in archive else archive[archive.files[0]]
        flag = np.zeros(reference.shape, dtype=bool)
        flag[frame["z"].to_numpy()[branch], frame["y"].to_numpy()[branch],
             frame["x"].to_numpy()[branch]] = True
        inside = reference != 0

        planes = (int(np.argmax(flag.sum(axis=(1, 2)))),
                  int(np.argmax(flag.sum(axis=(0, 2)))),
                  int(np.argmax(flag.sum(axis=(0, 1)))))
        figure = _figure(figsize=(15, 5.5))
        panels = figure.subplots(1, 3)
        for axes, (view, index) in zip(panels, zip(("axial", "coronal", "sagittal"), planes)):
            if view == "axial":
                base, mark, seg = reference[index], flag[index], inside[index]
            elif view == "coronal":
                base, mark, seg = (np.flipud(reference[:, index, :]), np.flipud(flag[:, index, :]),
                                   np.flipud(inside[:, index, :]))
            else:
                base, mark, seg = (np.flipud(reference[:, :, index]), np.flipud(flag[:, :, index]),
                                   np.flipud(inside[:, :, index]))
            axes.imshow(base, cmap="gray", interpolation="nearest",
                        vmax=float(np.percentile(base[base > 0], 99)) if (base > 0).any() else 1)
            axes.imshow(np.ma.masked_where(~seg, np.ones_like(base)), cmap="winter",
                        alpha=0.18, interpolation="nearest")
            axes.imshow(np.ma.masked_where(~mark, np.ones_like(base)), cmap="autumn",
                        alpha=0.95, interpolation="nearest")
            axes.set_title(f"{view} {index} — {int(mark.sum())} branch voxels", fontsize=9)
            axes.set_axis_off()
        figure.suptitle(f"{sample} / {seg_name}: high-rs voxels (red) on the first-echo image; "
                        f"segmentation tinted blue", fontsize=12)
        _save(figure, OUT / "spatial_maps" / f"{sample}_{seg_name}_branch.png", saved)
        del reference, flag, inside

    # ---- 7. constraint analysis ---------------------------------------
    print("7. constraint analysis")
    every_rs = np.concatenate([pools[l]["rs"] for l in labels])
    figure = _figure(figsize=(12, 5.5))
    left, right = figure.subplots(1, 2)
    left.hist(every_rs, bins=400, color="#666")
    left.set(yscale="log", xlabel="rs_selected (1/ms)", ylabel="voxels")
    left.axvline(BRANCH_RS, color="red", ls="--", label=f"{BRANCH_RS}")
    left.axvline(1 / 8.0, color="green", ls="--", label="1/TE_min = 0.125")
    left.set_title("All materials — is the branch separated?")
    left.legend()
    for label in labels:
        right.hist(pools[label]["rs"], bins=300, histtype="step", lw=1.3,
                   color=LABEL_COLORS[label], label=label, density=True)
    right.set(yscale="log", xlabel="rs_selected (1/ms)", ylabel="density")
    right.axvline(BRANCH_RS, color="black", ls="--")
    right.set_title("By material")
    right.legend(fontsize=7)
    _save(figure, OUT / "constraint_analysis" / "rs_distribution.png", saved)

    figure = _figure(figsize=(11, 5))
    axes = figure.subplots()
    for position, (sample, values) in enumerate(sorted(rs_by_sample.items()), start=1):
        axes.violinplot([values], positions=[position], showextrema=False, widths=0.8)
        axes.text(position, float(np.percentile(values, 99.99)),
                  f"max {values.max():.3f}", ha="center", fontsize=7)
    axes.set_xticks(range(1, len(rs_by_sample) + 1))
    axes.set_xticklabels(sorted(rs_by_sample))
    axes.axhline(BRANCH_RS, color="red", ls="--", label=f"branch cut {BRANCH_RS}")
    axes.set(ylabel="rs_selected (1/ms)")
    axes.set_title("rs by sample — complete voxel population")
    axes.legend()
    axes.grid(alpha=0.25, axis="y")
    _save(figure, OUT / "constraint_analysis" / "rs_by_sample.png", saved)

    # small diagnostic refit on branch voxels
    from scipy.optimize import curve_fit
    refit_rows = []
    for (sample, seg_name), store in curve_store.items():
        te = store["te"]
        for position in range(min(40, store["high_rs"].shape[0])):
            row = store["rows_high"].iloc[position]
            curve = store["high_rs"][position].astype(float)
            peak = float(curve.max())
            entry = {"sample": sample, "voxel": f"({int(row['x'])},{int(row['y'])},{int(row['z'])})",
                     "rs_B_unconstrained": float(row["rs_B"]),
                     "I0_B_unconstrained": float(row["I0_B"]),
                     "R2_B_unconstrained": float(row["R2_B"]),
                     "rs_A": float(row["rs_noB"]), "R2_A": float(row["R2_noB"])}
            for bound, tag in ((0.15, "0p15"), (0.125, "0p125")):
                try:
                    params, _ = curve_fit(model_with_baseline, te, curve,
                                          p0=[peak, min(float(row["rs_B"]), bound * 0.9),
                                              max(float(curve.min()), 0.0)],
                                          bounds=([0, 0, 0], [10 * peak, bound, peak]),
                                          maxfev=8000)
                    predicted = model_with_baseline(te, *params)
                    ss_res = float(np.sum((curve - predicted) ** 2))
                    ss_tot = float(np.sum((curve - curve.mean()) ** 2))
                    entry[f"rs_B_max{tag}"] = float(params[1])
                    entry[f"I0_B_max{tag}"] = float(params[0])
                    entry[f"R2_B_max{tag}"] = 1 - ss_res / ss_tot if ss_tot else 0.0
                except Exception:
                    entry[f"rs_B_max{tag}"] = np.nan
            refit_rows.append(entry)
    refit = pd.DataFrame(refit_rows)

    if len(refit):
        figure = _figure(figsize=(12, 5.5))
        left, right = figure.subplots(1, 2)
        left.scatter(refit["rs_B_unconstrained"], refit["rs_B_max0p15"], s=18,
                     color="#c9314e", label="bound 0.15")
        left.scatter(refit["rs_B_unconstrained"], refit["rs_A"], s=18,
                     color="#4a6fe3", label="Model A (B=0)")
        left.set(xlabel="rs, unconstrained Model B (1/ms)", ylabel="rs, refit (1/ms)")
        left.set_title("Where the branch voxels move when rs is bounded")
        left.legend(fontsize=8)
        left.grid(alpha=0.25)
        for column, colour, tag in (("R2_B_unconstrained", "#c9314e", "B unconstrained"),
                                    ("R2_B_max0p15", "#e8783c", "B, rs ≤ 0.15"),
                                    ("R2_A", "#4a6fe3", "Model A")):
            right.hist(refit[column].dropna(), bins=30, histtype="step", lw=1.6,
                       color=colour, label=f"{tag} (median {refit[column].median():.4f})")
        right.set(xlabel="R²", ylabel="voxels")
        right.set_title("Goodness of fit on the same branch voxels")
        right.legend(fontsize=8)
        right.grid(alpha=0.25)
        figure.suptitle(f"Diagnostic refit of {len(refit)} representative high-rs voxels "
                        "(no existing results modified)", fontsize=11)
        _save(figure, OUT / "constraint_analysis" / "refit_comparison.png", saved)

    # ---- summary -------------------------------------------------------
    branch_table.to_csv(OUT / "summary" / "branch_statistics.csv", index=False)
    refit.to_csv(OUT / "summary" / "refit_representative_voxels.csv", index=False)

    total_03 = sum(len(f) for f in frames_03.values())
    total_branch = sum(int((f["rs_sel"] > BRANCH_RS).sum()) for f in frames_03.values())
    all_branch = int((every_rs > BRANCH_RS).sum())

    lines = [
        "# High-rs branch — diagnostic summary", "",
        f"Working cut: `rs_selected > {BRANCH_RS}` 1/ms. Generated {time.strftime('%Y-%m-%d')}.",
        "Purely diagnostic; no existing fit result was modified.", "",
        "## How many voxels are affected", "",
        f"- **{all_branch:,} of {every_rs.size:,}** voxels overall ({100*all_branch/every_rs.size:.2f}%).",
        f"- Within 0.3 mg/mL: **{total_branch:,} of {total_03:,}** ({100*total_branch/total_03:.1f}%).", "",
        "| sample | segmentation | group | n | % of segmentation | median rs | median I0 | median B | median R² (B) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, row in branch_table.iterrows():
        lines.append(f"| {row['sample']} | {row['segmentation']} | {row['group']} | {row['n']:,} | "
                     f"{row['pct_of_segmentation']:.2f}% | {row['median_rs']:.4f} | "
                     f"{row['median_I0']:.0f} | {row['median_B']:.1f} | {row['median_R2_B']:.4f} |")
    lines += ["", "## Affected concentrations and samples", ""]
    for label in labels:
        count = int((pools[label]["rs"] > BRANCH_RS).sum())
        lines.append(f"- `{label}`: {count:,} of {pools[label]['rs'].size:,} "
                     f"({100*count/pools[label]['rs'].size:.2f}%)")
    lines += ["", "## rs distribution and candidate bounds", "",
              f"- Maximum observed rs: **{every_rs.max():.4f} 1/ms**.",
              f"- Highest physically expected material (0.3 mg/mL) median: "
              f"{np.median(pools['0.3']['rs']):.4f}, "
              f"99th percentile {np.percentile(pools['0.3']['rs'], 99):.4f}.",
              f"- Voxels between 0.13 and 0.20: {int(((every_rs > 0.13) & (every_rs < 0.20)).sum()):,} "
              "(the sparser the gap, the cleaner the separation).", ""]
    if len(refit):
        lines += ["## Diagnostic refit of representative branch voxels", "",
                  f"{len(refit)} voxels refitted three ways:", "",
                  "| fit | median rs | median R² |", "|---|---|---|",
                  f"| Model B, unconstrained | {refit['rs_B_unconstrained'].median():.4f} | "
                  f"{refit['R2_B_unconstrained'].median():.4f} |",
                  f"| Model B, rs ≤ 0.15 | {refit['rs_B_max0p15'].median():.4f} | "
                  f"{refit['R2_B_max0p15'].median():.4f} |",
                  f"| Model B, rs ≤ 0.125 | {refit['rs_B_max0p125'].median():.4f} | "
                  f"{refit['R2_B_max0p125'].median():.4f} |",
                  f"| Model A (B = 0) | {refit['rs_A'].median():.4f} | {refit['R2_A'].median():.4f} |", ""]
    (OUT / "summary" / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    saved.append(OUT / "summary" / "summary.md")

    print(f"\nPNGs: {len([p for p in saved if p.suffix == '.png'])}  "
          f"elapsed {time.time()-started:.0f}s")
    print(f"Output: {OUT}")
    print("\n--- branch table ---")
    print(branch_table.to_string(index=False))
    if len(refit):
        print("\n--- refit medians ---")
        for column in ("rs_B_unconstrained", "rs_B_max0p15", "rs_B_max0p125", "rs_A"):
            print(f"  {column:22s} rs median {refit[column].median():.5f}")
        for column in ("R2_B_unconstrained", "R2_B_max0p15", "R2_B_max0p125", "R2_A"):
            if column in refit:
                print(f"  {column:22s} R2 median {refit[column].median():.5f}")
    print(f"\nmax rs overall: {every_rs.max():.4f}   "
          f"voxels in 0.13-0.20 gap: {int(((every_rs>0.13)&(every_rs<0.20)).sum()):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
