#!/usr/bin/env python3
"""Compare I0 normalisation strategies before the classification stage.

Diagnostic and preparatory only: reads the existing AIC-selected fits, writes
normalised copies and figures into new directories. No fit is repeated and no
existing result is modified.

Strategies
    A  per-sample 0-SPION reference (Syringes, 3D, Pill2 only; Pill1 has none)
    B  a transferable multiplicative scale, tested against the data before use
    C  tissue-referenced, for Pill1, as a diagnostic - tissue is NOT 0 SPION
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_ROOT  # noqa: E402

import numpy as np  # noqa: E402

MRI = DATA_ROOT
FIT_ROOT = MRI / "i0_rs" / "Segmentations"
SEG_ROOT = MRI / "Segmentations"
NORM_ROOT = MRI / "i0_rs" / "normalized"
FIG_ROOT = MRI / "i0_rs" / "model_diagnostics" / "normalization"

SEED = 42
MAX_PLOT = 120_000
#: Voxels above this rate are flagged; see the high_rs_artifact diagnostic.
HIGH_RS = 0.15
#: Model B has three parameters, so fewer than four informative echoes makes it
#: unidentifiable.
MIN_INFORMATIVE = 4
#: An echo counts as informative when it exceeds this multiple of the voxel's
#: own noise floor.
NOISE_FACTOR = 2.0
#: Samples reserved for training/calibration. Pill2 is the held-out test set.
TRAINING = ("Syringes", "3D", "Pill1")
CONC = {"0": 0.0, "0.05": 0.05, "0.1": 0.1, "0.2": 0.2, "0.25": 0.25, "0.3": 0.3}


def main() -> int:
    import echoviewer  # noqa: F401
    import pandas as pd
    from echoviewer.diagnostics import (LABEL_COLORS, LABEL_ORDER, _figure, _save,
                                        bhattacharyya_overlap, iqr, iter_fit_dirs,
                                        load_voxel_curves, read_voxels, subsample)

    started = time.time()
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    NEEDED = ("x", "y", "z", "I0_noB", "rs_noB", "I0_B", "rs_B", "B",
              "model_selected", "fit_success")

    # ---- load, derive selected representation, flag quality -------------
    print("Reading fits and counting informative echoes")
    frames: dict[tuple[str, str], "pd.DataFrame"] = {}
    for sample, seg_name, label, seg_dir in iter_fit_dirs(FIT_ROOT):
        frame = read_voxels(seg_dir, NEEDED)
        chose_a = frame["model_selected"].to_numpy() == 1
        frame = frame.assign(
            sample=sample, segmentation=seg_name, material=label,
            rs_selected=np.where(chose_a, frame["rs_noB"], frame["rs_B"]).astype(np.float32),
            I0_selected=np.where(chose_a, frame["I0_noB"], frame["I0_B"]).astype(np.float32),
            B_selected=np.where(chose_a, 0.0, frame["B"]).astype(np.float32),
        )

        # Informative echoes, measured from the data rather than the model.
        coords = list(zip(frame["z"].to_numpy(np.intp), frame["y"].to_numpy(np.intp),
                          frame["x"].to_numpy(np.intp)))
        _, curves = load_voxel_curves(SEG_ROOT / sample / "volumes" / seg_name, coords)
        floor = np.median(curves[:, -5:], axis=1, keepdims=True)
        n_informative = (curves > NOISE_FACTOR * np.maximum(floor, 1e-6)).sum(axis=1)
        del curves

        frame = frame.assign(
            n_informative=n_informative.astype(np.int16),
            flag_high_rs=(frame["rs_selected"].to_numpy() > HIGH_RS).astype(np.int8),
            flag_low_information=(n_informative < MIN_INFORMATIVE).astype(np.int8),
        )
        frame = frame.assign(
            quality_ok=((frame["flag_high_rs"] == 0) &
                        (frame["flag_low_information"] == 0)).astype(np.int8)
        )
        frames[(sample, seg_name)] = frame
        print(f"  {sample}/{seg_name} ({label}) {len(frame):,} voxels, "
              f"{int(frame['flag_high_rs'].sum()):,} high-rs, "
              f"{int(frame['flag_low_information'].sum()):,} low-information")

    samples = sorted({s for s, _ in frames})
    labels_present = [l for l in LABEL_ORDER
                      if any(f["material"].iloc[0] == l for f in frames.values())]

    def pool(sample=None, material=None, column="I0_selected", quality=False):
        parts = []
        for (s, _), frame in frames.items():
            if sample and s != sample:
                continue
            if material and frame["material"].iloc[0] != material:
                continue
            block = frame[frame["quality_ok"] == 1] if quality else frame
            parts.append(block[column].to_numpy(np.float32))
        return np.concatenate(parts) if parts else np.zeros(0, np.float32)

    # ---- Strategy A: per-sample 0-SPION reference -----------------------
    print("\nStrategy A: per-sample 0-SPION reference")
    reference: dict[str, dict] = {}
    for sample in samples:
        zero = pool(sample=sample, material="0", quality=True)
        if zero.size == 0:
            print(f"  {sample:9s} no 0-SPION segmentation — not normalised under A")
            continue
        reference[sample] = {
            "reference_material": "0",
            "reference_segmentation": next(
                seg for (s, seg), f in frames.items()
                if s == sample and f["material"].iloc[0] == "0"),
            "n_voxels_used": int(zero.size),
            "median_I0_reference": float(np.median(zero)),
            "note": "quality_ok voxels only (not high-rs, >=4 informative echoes)",
        }
        print(f"  {sample:9s} median I0(0 SPION) = {np.median(zero):8.1f} "
              f"from {zero.size:,} quality voxels")

    # ---- Strategy B: is the difference multiplicative? ------------------
    print("\nStrategy B: testing a multiplicative sample scale")
    median_table = {}
    for sample in samples:
        for label in labels_present:
            values = pool(sample=sample, material=label, column="I0_selected", quality=True)
            if values.size:
                median_table[(sample, label)] = float(np.median(values))

    # log I0(sample, conc) = a_sample + b_conc, solved by least squares on the
    # concentration entries only. If the residuals are small the difference
    # between phantoms really is one multiplicative number.
    conc_labels = [l for l in labels_present if l in CONC]
    rows = [(s, l) for (s, l) in median_table if l in conc_labels]
    sample_index = {s: i for i, s in enumerate(sorted({s for s, _ in rows}))}
    conc_index = {l: i for i, l in enumerate(conc_labels)}
    design = np.zeros((len(rows), len(sample_index) + len(conc_index)))
    target = np.zeros(len(rows))
    for r, (s, l) in enumerate(rows):
        design[r, sample_index[s]] = 1
        design[r, len(sample_index) + conc_index[l]] = 1
        target[r] = np.log(median_table[(s, l)])
    # Fix the gauge: the first concentration effect is zero.
    design = np.delete(design, len(sample_index), axis=1)
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    predicted = design @ coefficients
    residual = target - predicted
    scales = {s: float(np.exp(coefficients[i])) for s, i in sample_index.items()}
    conc_effect = {conc_labels[0]: 1.0}
    for l, i in conc_index.items():
        if i > 0:
            conc_effect[l] = float(np.exp(coefficients[len(sample_index) + i - 1]))

    rms = float(np.sqrt(np.mean(residual ** 2)))
    max_dev = float(np.max(np.abs(np.exp(residual) - 1)))
    print(f"  two-way log fit: RMS residual {rms:.4f} in log units "
          f"(max multiplicative deviation {100*max_dev:.1f}%)")
    for s, value in sorted(scales.items()):
        print(f"    scale[{s:9s}] = {value:8.1f}")
    print("  concentration effect (relative to "
          f"{conc_labels[0]}): " + ", ".join(f"{l}:{v:.3f}" for l, v in sorted(conc_effect.items())))

    # ---- Strategy C: tissue reference for Pill1 (diagnostic) ------------
    print("\nStrategy C: tissue-referenced (Pill1, diagnostic only)")
    tissue_reference = {}
    for sample in samples:
        tissue = pool(sample=sample, material="tissue", quality=True)
        if tissue.size:
            tissue_reference[sample] = float(np.median(tissue))
            print(f"  {sample:9s} median I0(tissue) = {tissue_reference[sample]:8.1f} "
                  f"({tissue.size:,} voxels)")

    # ---- apply and save --------------------------------------------------
    print("\nWriting normalised copies")
    metadata = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed": SEED,
                "quality_criteria": {
                    "flag_high_rs": f"rs_selected > {HIGH_RS} 1/ms",
                    "flag_low_information": f"fewer than {MIN_INFORMATIVE} echoes above "
                                            f"{NOISE_FACTOR}x the voxel noise floor",
                    "quality_ok": "neither flag set"},
                "strategies": {}, "notes": []}

    def write_strategy(name, factor_for, note):
        entries = {}
        for sample in samples:
            factor = factor_for(sample)
            if factor is None:
                continue
            parts = [f for (s, _), f in frames.items() if s == sample]
            block = pd.concat(parts, ignore_index=True)
            block = block.assign(
                I0_normalized=(block["I0_selected"] / factor).astype(np.float32),
                normalization_factor=np.float32(factor),
            )
            folder = NORM_ROOT / name / sample
            folder.mkdir(parents=True, exist_ok=True)
            keep = ["sample", "segmentation", "material", "x", "y", "z",
                    "rs_selected", "I0_selected", "I0_normalized", "B_selected",
                    "model_selected", "n_informative", "flag_high_rs",
                    "flag_low_information", "quality_ok", "normalization_factor"]
            block[keep].to_csv(folder / "voxels.csv.gz", index=False,
                               float_format="%.6g", compression="gzip")
            entry = {"sample": sample, "normalization_factor": float(factor),
                     "n_voxels": int(len(block)),
                     "n_quality_ok": int(block["quality_ok"].sum()),
                     "method": note}
            entry.update(reference.get(sample, {}) if name == "strategy_A" else {})
            (folder / "metadata.json").write_text(json.dumps(entry, indent=2))
            entries[sample] = entry
            del block, parts
        metadata["strategies"][name] = entries
        return entries

    write_strategy(
        "strategy_A",
        lambda s: reference[s]["median_I0_reference"] if s in reference else None,
        "I0_selected / median(I0_selected of that sample's 0-SPION segmentation)")
    write_strategy(
        "strategy_B",
        lambda s: scales.get(s),
        "I0_selected / sample scale from a two-way log-additive fit of "
        "median I0 over (sample, concentration)")
    write_strategy(
        "strategy_C",
        lambda s: tissue_reference[s] if s == "Pill1" else None,
        "DIAGNOSTIC ONLY: I0_selected / median(I0_selected in tissue). "
        "Tissue is NOT 0 mg/mL SPION.")

    metadata["strategy_B_evidence"] = {
        "model": "log median I0(sample, concentration) = a_sample + b_concentration",
        "rms_log_residual": rms,
        "max_multiplicative_deviation": max_dev,
        "sample_scales": scales,
        "concentration_effects": conc_effect,
        "interpretation": ("small residuals support one multiplicative number per "
                           "phantom; large residuals mean the phantoms differ in a "
                           "concentration-dependent way and no single scale transfers"),
    }
    metadata["notes"].append(
        "Pill1 has no 0-SPION segmentation, so strategy A leaves it unnormalised.")
    metadata["notes"].append(
        "Pill2's 0-SPION region is used only as an unsupervised intensity reference; "
        "its SPION labels were not used to choose or tune any normalisation.")
    (FIG_ROOT / "normalization_metadata.json").write_text(json.dumps(metadata, indent=2))
    saved.append(FIG_ROOT / "normalization_metadata.json")

    # ---- statistics ------------------------------------------------------
    print("\nStatistics")
    stat_rows = []
    for (sample, seg_name), frame in frames.items():
        label = frame["material"].iloc[0]
        for strategy, factor in (("raw", 1.0),
                                 ("strategy_A", reference.get(sample, {}).get("median_I0_reference")),
                                 ("strategy_B", scales.get(sample)),
                                 ("strategy_C", tissue_reference.get(sample) if sample == "Pill1" else None)):
            if factor is None:
                continue
            values = frame["I0_selected"].to_numpy(np.float32) / factor
            q1, q3 = iqr(values)
            rq1, rq3 = iqr(frame["rs_selected"].to_numpy())
            stat_rows.append({
                "strategy": strategy, "sample": sample, "segmentation": seg_name,
                "material": label, "n_voxels": len(frame),
                "n_quality_ok": int(frame["quality_ok"].sum()),
                "median_I0": float(np.median(values)), "IQR_I0": q3 - q1, "std_I0": float(values.std()),
                "median_rs": float(np.median(frame["rs_selected"])), "IQR_rs": rq3 - rq1,
                "std_rs": float(frame["rs_selected"].std()),
                "median_B": float(np.median(frame["B_selected"])),
            })
    stats = pd.DataFrame(stat_rows)
    stats.to_csv(FIG_ROOT / "normalization_statistics.csv", index=False)
    saved.append(FIG_ROOT / "normalization_statistics.csv")

    # ---- feature-space separability (training samples only) --------------
    print("Feature-space overlap (Syringes + 3D + Pill1 only; Pill2 labels unused)")
    def training_pool(material, columns, strategy):
        parts = []
        for (s, _), frame in frames.items():
            if s not in TRAINING or frame["material"].iloc[0] != material:
                continue
            factor = {"raw": 1.0,
                      "strategy_A": reference.get(s, {}).get("median_I0_reference"),
                      "strategy_B": scales.get(s)}[strategy]
            if factor is None:
                return None
            block = frame[frame["quality_ok"] == 1]
            data = {"rs": block["rs_selected"].to_numpy(np.float64),
                    "I0": block["I0_selected"].to_numpy(np.float64) / factor}
            parts.append(np.column_stack([data[c] for c in columns]))
        return np.vstack(parts) if parts else None

    pairs = [("0", "0.05"), ("0.05", "0.1"), ("tissue", "0.05"), ("tissue", "0.1"),
             ("0.1", "0.2"), ("0.2", "0.25"), ("0.25", "0.3")]
    feature_sets = [("rs alone", ("rs",), "raw"),
                    ("rs + I0 raw", ("rs", "I0"), "raw"),
                    ("rs + I0 strategy_A", ("rs", "I0"), "strategy_A"),
                    ("rs + I0 strategy_B", ("rs", "I0"), "strategy_B")]
    overlap_rows = []
    for name, columns, strategy in feature_sets:
        for first, second in pairs:
            a = training_pool(first, columns, strategy)
            b = training_pool(second, columns, strategy)
            value = bhattacharyya_overlap(a, b) if a is not None and b is not None else np.nan
            overlap_rows.append({"feature_set": name, "class_a": first,
                                 "class_b": second, "overlap": value})
    overlaps = pd.DataFrame(overlap_rows)
    overlaps.to_csv(FIG_ROOT / "feature_overlap.csv", index=False)
    saved.append(FIG_ROOT / "feature_overlap.csv")

    # ---- figures ---------------------------------------------------------
    print("Figures")
    strategies_for_plot = [("raw", "Raw I0_selected", lambda s: 1.0),
                           ("strategy_A", "Strategy A (0-SPION reference)",
                            lambda s: reference.get(s, {}).get("median_I0_reference")),
                           ("strategy_B", "Strategy B (transferable scale)",
                            lambda s: scales.get(s))]

    def scatter(axes, strategy_factor, sample=None, limits=None):
        rng = np.random.default_rng(SEED)
        total = drawn = 0
        for label in labels_present:
            xs, ys = [], []
            for (s, _), frame in frames.items():
                if (sample and s != sample) or frame["material"].iloc[0] != label:
                    continue
                factor = strategy_factor(s)
                if factor is None:
                    continue
                xs.append(frame["rs_selected"].to_numpy(np.float32))
                ys.append(frame["I0_selected"].to_numpy(np.float32) / factor)
            if not xs:
                continue
            x = np.concatenate(xs)
            y = np.concatenate(ys)
            total += x.size
            x, y = subsample([x, y], MAX_PLOT, rng)
            drawn += x.size
            axes.scatter(x, y, s=2, alpha=0.12, linewidths=0,
                         color=LABEL_COLORS[label], label=label)
        axes.set_xlabel("rs_selected (1/ms)")
        axes.grid(alpha=0.25)
        if limits:
            axes.set_xlim(*limits[0])
            axes.set_ylim(*limits[1])
        return total, drawn

    # A/B: I0 distributions raw and normalised
    for tag, title, factor_for, name in (
        ("raw", "Raw I0_selected", lambda s: 1.0, "raw_I0_distributions.png"),
        ("strategy_A", "Strategy A normalised I0",
         lambda s: reference.get(s, {}).get("median_I0_reference"),
         "normalized_I0_distributions.png"),
    ):
        figure = _figure(figsize=(13, 6))
        axes = figure.subplots()
        position = 0
        ticks, tick_labels = [], []
        for sample in samples:
            for label in labels_present:
                values = []
                for (s, _), frame in frames.items():
                    if s == sample and frame["material"].iloc[0] == label:
                        factor = factor_for(s)
                        if factor is None:
                            continue
                        values.append(frame["I0_selected"].to_numpy(np.float32) / factor)
                if not values:
                    continue
                position += 1
                data = np.concatenate(values)
                parts = axes.violinplot([data], positions=[position], showextrema=False, widths=0.8)
                parts["bodies"][0].set_facecolor(LABEL_COLORS[label])
                parts["bodies"][0].set_alpha(0.7)
                q1, q3 = iqr(data)
                axes.vlines(position, q1, q3, color="black", lw=3)
                ticks.append(position)
                tick_labels.append(f"{sample}\n{label}")
            position += 0.8
        axes.set_xticks(ticks)
        axes.set_xticklabels(tick_labels, fontsize=6, rotation=90)
        axes.set_ylabel("I0" + ("" if tag == "raw" else " / reference"))
        axes.set_title(f"{title} — every valid voxel, median and IQR marked")
        if tag != "raw":
            axes.axhline(1.0, color="black", ls="--", lw=1)
        axes.grid(alpha=0.25, axis="y")
        _save(figure, FIG_ROOT / name, saved)

    # C/D: rs vs I0, all samples and per sample, raw and normalised
    for tag, title, factor_for, all_name, panel_name in (
        ("raw", "Raw I0_selected", lambda s: 1.0,
         "raw_I0_vs_rs_all_samples.png", "raw_I0_vs_rs_by_sample.png"),
        ("strategy_A", "Strategy A normalised",
         lambda s: reference.get(s, {}).get("median_I0_reference"),
         "normalized_I0_vs_rs_all_samples.png", "normalized_I0_vs_rs_by_sample.png"),
    ):
        figure = _figure(figsize=(10, 8))
        axes = figure.subplots()
        total, drawn = scatter(axes, factor_for)
        axes.set_ylabel("I0_selected" if tag == "raw" else "I0_normalized")
        axes.set_title(f"{title} — all samples\n{drawn:,} of {total:,} plotted, seed {SEED}")
        legend = axes.legend(markerscale=6, fontsize=8)
        for handle in legend.legend_handles:
            handle.set_alpha(1.0)
        _save(figure, FIG_ROOT / all_name, saved)

        limits = (axes.get_xlim(), axes.get_ylim())
        figure = _figure(figsize=(14, 11))
        panels = figure.subplots(2, 2)
        for sub, sample in zip(panels.ravel(), samples):
            total, drawn = scatter(sub, factor_for, sample=sample, limits=limits)
            sub.set_ylabel("I0_selected" if tag == "raw" else "I0_normalized")
            sub.set_title(f"{sample} — {total:,} voxels" +
                          ("" if factor_for(sample) is not None else "  (no reference)"),
                          fontsize=10)
            if total:
                legend = sub.legend(markerscale=6, fontsize=7)
                for handle in legend.legend_handles:
                    handle.set_alpha(1.0)
        figure.suptitle(f"{title} — identical axes, seed {SEED}", fontsize=13)
        _save(figure, FIG_ROOT / panel_name, saved)

    # E: normalised distributions per material
    figure = _figure(figsize=(11, 5.5))
    axes = figure.subplots()
    for position, label in enumerate(labels_present, start=1):
        values = []
        for (s, _), frame in frames.items():
            factor = reference.get(s, {}).get("median_I0_reference")
            if factor is None or frame["material"].iloc[0] != label:
                continue
            values.append(frame["I0_selected"].to_numpy(np.float32) / factor)
        if not values:
            continue
        data = np.concatenate(values)
        parts = axes.violinplot([data], positions=[position], showextrema=False, widths=0.8)
        parts["bodies"][0].set_facecolor(LABEL_COLORS[label])
        q1, q3 = iqr(data)
        axes.vlines(position, q1, q3, color="black", lw=3)
        axes.plot(position, np.median(data), "o", color="white", markeredgecolor="black")
    axes.set_xticks(range(1, len(labels_present) + 1))
    axes.set_xticklabels(labels_present)
    axes.axhline(1.0, color="black", ls="--", lw=1)
    axes.set_ylabel("I0_normalized (strategy A)")
    axes.set_title("Normalised I0 by material — does it still carry concentration information?")
    axes.grid(alpha=0.25, axis="y")
    _save(figure, FIG_ROOT / "concentration_distributions.png", saved)

    # F: feature space per strategy, all + per sample
    for tag, title, factor_for in strategies_for_plot:
        figure = _figure(figsize=(10, 8))
        axes = figure.subplots()
        total, drawn = scatter(axes, factor_for)
        axes.set_ylabel("I0" + ("" if tag == "raw" else "_normalized"))
        axes.set_title(f"Feature space — {title}\n{drawn:,} of {total:,} plotted, seed {SEED}")
        legend = axes.legend(markerscale=6, fontsize=8)
        for handle in legend.legend_handles:
            handle.set_alpha(1.0)
        _save(figure, FIG_ROOT / f"feature_space_rs_vs_I0_{tag}.png", saved)
    for sample in samples:
        figure = _figure(figsize=(9, 7.5))
        axes = figure.subplots()
        total, drawn = scatter(axes, lambda s: reference.get(s, {}).get("median_I0_reference"),
                               sample=sample)
        if total == 0:
            total, drawn = scatter(axes, lambda s: scales.get(s), sample=sample)
            axes.set_title(f"{sample} — strategy B scale (no 0-SPION reference)\n"
                           f"{drawn:,} of {total:,} plotted, seed {SEED}")
        else:
            axes.set_title(f"{sample} — strategy A\n{drawn:,} of {total:,} plotted, seed {SEED}")
        axes.set_ylabel("I0_normalized")
        legend = axes.legend(markerscale=6, fontsize=8)
        for handle in legend.legend_handles:
            handle.set_alpha(1.0)
        _save(figure, FIG_ROOT / f"feature_space_rs_vs_I0_normalized_{sample}.png", saved)

    # normalization_comparison: overlap per feature set
    figure = _figure(figsize=(12, 6))
    axes = figure.subplots()
    width = 0.2
    names = [n for n, _, _ in feature_sets]
    positions = np.arange(len(pairs))
    for offset, name in enumerate(names):
        values = [overlaps[(overlaps.feature_set == name) &
                           (overlaps.class_a == a) & (overlaps.class_b == b)]["overlap"].iloc[0]
                  for a, b in pairs]
        axes.bar(positions + (offset - 1.5) * width, values, width, label=name)
    axes.set_xticks(positions)
    axes.set_xticklabels([f"{a}\nvs {b}" for a, b in pairs], fontsize=8)
    axes.set_ylabel("class overlap (lower is better)")
    axes.set_title("Does normalising I0 improve separation?  Training samples only "
                   "(Syringes + 3D + Pill1); Pill2 labels not used")
    axes.legend(fontsize=8)
    axes.grid(alpha=0.25, axis="y")
    _save(figure, FIG_ROOT / "normalization_comparison.png", saved)

    # ---- report ----------------------------------------------------------
    print("\n" + "=" * 76)
    print("REPORT")
    print("=" * 76)
    print("\n1. Raw phantom-to-phantom I0 differences (median over all voxels)")
    for sample in samples:
        values = pool(sample=sample)
        print(f"   {sample:9s} median I0 {np.median(values):8.1f}  n={values.size:,}")
    medians = [float(np.median(pool(sample=s))) for s in samples]
    print(f"   spread: {max(medians)/min(medians):.2f}x between the brightest and dimmest")

    print("\n2-3. Strategy A / B factors")
    for sample in samples:
        a = reference.get(sample, {}).get("median_I0_reference")
        print(f"   {sample:9s} A: {a if a else 'NO 0-SPION REFERENCE':>22} "
              f"  B scale: {scales.get(sample, float('nan')):8.1f}")
    print(f"   multiplicative model RMS log residual {rms:.4f}; "
          f"largest deviation {100*max_dev:.1f}%")

    print("\n5. Median normalised I0 by material (strategy A; 1.0 = the 0-SPION reference)")
    for label in labels_present:
        values = []
        for (s, _), frame in frames.items():
            factor = reference.get(s, {}).get("median_I0_reference")
            if factor and frame["material"].iloc[0] == label:
                values.append(frame["I0_selected"].to_numpy(np.float32) / factor)
        if values:
            data = np.concatenate(values)
            q1, q3 = iqr(data)
            print(f"   {label:7s} median {np.median(data):6.3f}  IQR [{q1:.3f}, {q3:.3f}]")

    print("\n6/8. Class overlap by feature set (training samples only, lower is better)")
    print(f"   {'pair':16s}" + "".join(f"{n:>22s}" for n in names))
    for a, b in pairs:
        line = f"   {a + ' vs ' + b:16s}"
        for name in names:
            value = overlaps[(overlaps.feature_set == name) & (overlaps.class_a == a) &
                             (overlaps.class_b == b)]["overlap"].iloc[0]
            line += f"{value:22.3f}" if np.isfinite(value) else f"{'n/a':>22s}"
        print(line)

    print("\n12. Quality flags")
    total_v = sum(len(f) for f in frames.values())
    high = sum(int(f["flag_high_rs"].sum()) for f in frames.values())
    low = sum(int(f["flag_low_information"].sum()) for f in frames.values())
    ok = sum(int(f["quality_ok"].sum()) for f in frames.values())
    print(f"   {total_v:,} voxels: {high:,} high-rs ({100*high/total_v:.2f}%), "
          f"{low:,} low-information ({100*low/total_v:.2f}%), "
          f"{ok:,} pass both ({100*ok/total_v:.1f}%)")
    for label in labels_present:
        parts = [f for f in frames.values() if f["material"].iloc[0] == label]
        n = sum(len(f) for f in parts)
        flagged = sum(int((f["quality_ok"] == 0).sum()) for f in parts)
        print(f"     {label:7s} {100*flagged/n:5.1f}% flagged")

    print(f"\nPNGs: {len([p for p in saved if p.suffix == '.png'])}   "
          f"elapsed {time.time()-started:.0f}s")
    print(f"Normalised data: {NORM_ROOT}")
    print(f"Figures: {FIG_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
