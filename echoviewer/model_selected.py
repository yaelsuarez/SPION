"""The AIC-selected parameter representation, and its I0-vs-rs visualisations.

Model A is Model B with ``B = 0``, so the two are not rivals: AIC decides, per
voxel, whether the extra baseline parameter is supported. This module builds a
third, unified representation from the fits that already exist::

    if AIC_A <= AIC_B:   I0_selected = I0_A,  rs_selected = rs_A,  B_selected = 0
    else:                I0_selected = I0_B,  rs_selected = rs_B,  B_selected = B

The distinction that matters: when Model A wins, its *own* fitted parameters
are used. Taking Model B's ``I0`` and ``rs`` and forcing ``B`` to zero would
give a curve nobody fitted, and would carry the baseline's influence into the
other two parameters.

Nothing here refits and nothing here writes into the fit results. The stored
``model_selected`` column is used as the authority: it was computed at full
precision during fitting, whereas the CSV's AIC values are rounded to six
significant figures, which flips a handful of near-ties.

Naming: the fit outputs call Model A ``noB``. ``I0_noB`` is ``I0_A``,
``rs_noB`` is ``rs_A``, ``AIC_noB`` is ``AIC_A``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .diagnostics import (
    LABEL_COLORS,
    LABEL_ORDER,
    SEED,
    _figure,
    _save,
    iter_fit_dirs,
    read_voxels,
    subsample,
)

#: Samples in panel order.
SAMPLE_ORDER = ("3D", "Syringes", "Pill1", "Pill2")

#: Colours for the by-model view.
MODEL_COLORS = {"A": "#4a6fe3", "B": "#c9314e"}

#: Cap on plotted points per material in any one panel.
MAX_PER_CLASS = 120_000

#: Columns needed from each voxel CSV.
NEEDED = ("I0_noB", "rs_noB", "I0_B", "rs_B", "B", "AIC_noB", "AIC_B",
          "model_selected", "fit_success")


@dataclass
class PlotRecord:
    """Provenance for one figure: how many voxels existed, how many were drawn."""

    name: str
    available: int
    plotted: int
    seed: int = SEED

    @property
    def fraction(self) -> float:
        return self.plotted / self.available if self.available else 0.0

    def caption(self) -> str:
        return (f"{self.plotted:,} of {self.available:,} voxels plotted "
                f"({100 * self.fraction:.1f}%), random seed {self.seed}. "
                "Subsampling affects the display only.")


@dataclass
class SelectedData:
    """Per-(sample, material) arrays of all three representations."""

    groups: dict[tuple[str, str], dict[str, np.ndarray]] = field(default_factory=dict)

    def add(self, sample: str, label: str, frame) -> None:
        """Derive the selected representation for one segmentation and store it."""
        chose_a = frame["model_selected"].to_numpy() == 1

        rs_a = frame["rs_noB"].to_numpy(np.float32)
        i0_a = frame["I0_noB"].to_numpy(np.float32)
        rs_b = frame["rs_B"].to_numpy(np.float32)
        i0_b = frame["I0_B"].to_numpy(np.float32)
        baseline = frame["B"].to_numpy(np.float32)

        entry = self.groups.setdefault((sample, label), {})
        block = {
            "rs_A": rs_a, "I0_A": i0_a,
            "rs_B": rs_b, "I0_B": i0_b, "B": baseline,
            # Model A's own parameters when A wins - not Model B's with B zeroed.
            "rs_sel": np.where(chose_a, rs_a, rs_b),
            "I0_sel": np.where(chose_a, i0_a, i0_b),
            "B_sel": np.where(chose_a, np.float32(0.0), baseline),
            "chose_a": chose_a.astype(np.int8),
        }
        for key, values in block.items():
            entry.setdefault(key, []).append(values)

    def finish(self) -> None:
        """Concatenate each group's chunks once."""
        for key, entry in self.groups.items():
            self.groups[key] = {name: np.concatenate(parts) for name, parts in entry.items()}

    # -- views -------------------------------------------------------------

    @property
    def labels(self) -> list[str]:
        present = {label for _, label in self.groups}
        return [name for name in LABEL_ORDER if name in present]

    @property
    def samples(self) -> list[str]:
        present = {sample for sample, _ in self.groups}
        return [name for name in SAMPLE_ORDER if name in present] + \
               sorted(present - set(SAMPLE_ORDER))

    def by_label(self, label: str, sample: str | None = None) -> dict[str, np.ndarray]:
        """Every voxel of one material, optionally restricted to one sample."""
        parts = [values for (s, l), values in self.groups.items()
                 if l == label and (sample is None or s == sample)]
        if not parts:
            return {}
        return {name: np.concatenate([p[name] for p in parts]) for name in parts[0]}

    def count(self, sample: str | None = None) -> int:
        return sum(v["rs_sel"].size for (s, _), v in self.groups.items()
                   if sample is None or s == sample)


def collect(fit_root: Path, progress=print) -> SelectedData:
    """Read every segmentation once and build the selected representation."""
    data = SelectedData()
    entries = list(iter_fit_dirs(fit_root))
    for index, (sample, seg_name, label, seg_dir) in enumerate(entries, start=1):
        frame = read_voxels(seg_dir, NEEDED)
        progress(f"  [{index}/{len(entries)}] {sample}/{seg_name} ({label}) {len(frame):,} voxels")
        data.add(sample, label, frame)
        del frame
    data.finish()
    return data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def axis_limits(data: SelectedData) -> tuple[tuple[float, float], tuple[float, float]]:
    """Shared axis limits so all three representations are visually comparable."""
    rs_parts, i0_parts = [], []
    for values in data.groups.values():
        for key in ("rs_A", "rs_B", "rs_sel"):
            rs_parts.append(np.percentile(values[key], [0.2, 99.8]))
        for key in ("I0_A", "I0_B", "I0_sel"):
            i0_parts.append(np.percentile(values[key], [0.2, 99.8]))
    rs = np.vstack(rs_parts)
    i0 = np.vstack(i0_parts)
    return (float(rs[:, 0].min()), float(rs[:, 1].max())), \
           (float(i0[:, 0].min()), float(i0[:, 1].max()))


def _draw_by_material(axes, data, rs_key, i0_key, rng, sample=None, limits=None):
    """Scatter one representation, coloured by material. Returns (available, plotted)."""
    available = plotted = 0
    for label in data.labels:
        pool = data.by_label(label, sample=sample)
        if not pool:
            continue
        available += pool[rs_key].size
        x, y = subsample([pool[rs_key], pool[i0_key]], MAX_PER_CLASS, rng)
        plotted += x.size
        axes.scatter(x, y, s=2, alpha=0.12, linewidths=0,
                     color=LABEL_COLORS[label], label=f"{label} (n={pool[rs_key].size:,})")
        del pool
    axes.set_xlabel("rs (1/ms)")
    axes.set_ylabel("I0 (a.u.)")
    axes.grid(alpha=0.25)
    if limits:
        axes.set_xlim(*limits[0])
        axes.set_ylim(*limits[1])
    return available, plotted


def _legend(axes, **kwargs):
    legend = axes.legend(markerscale=6, fontsize=8, framealpha=0.9, **kwargs)
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
    return legend


def figures(data: SelectedData, out: Path, saved: list, progress=print) -> list[PlotRecord]:
    """Every figure requested for the model-selected analysis."""
    folder = Path(out) / "parameter_relationships"
    folder.mkdir(parents=True, exist_ok=True)
    limits = axis_limits(data)
    records: list[PlotRecord] = []

    # 1-3. the three representations, one figure each
    for rs_key, i0_key, name, title in (
        ("rs_A", "I0_A", "I0_vs_rs_Model_A_all_samples.png",
         "Model A:  I(TE) = I0·exp(−rs·TE)"),
        ("rs_B", "I0_B", "I0_vs_rs_Model_B_all_samples.png",
         "Model B:  I(TE) = I0·exp(−rs·TE) + B"),
        ("rs_sel", "I0_sel", "I0_vs_rs_Model_Selected_all_samples.png",
         "Model-selected (AIC):  Model A parameters where A wins, Model B where B wins"),
    ):
        figure = _figure(figsize=(10, 8.5))
        axes = figure.subplots()
        rng = np.random.default_rng(SEED)
        available, plotted = _draw_by_material(axes, data, rs_key, i0_key, rng, limits=limits)
        record = PlotRecord(name, available, plotted)
        records.append(record)
        axes.set_title(f"{title}\nall four samples, coloured by material\n{record.caption()}",
                       fontsize=10)
        _legend(axes)
        _save(figure, folder / name, saved)
        progress(f"  {name}")

    # 4. selected, coloured by which model won
    figure = _figure(figsize=(10, 8.5))
    axes = figure.subplots()
    rng = np.random.default_rng(SEED)
    available = plotted = 0
    for tag, wanted in (("A", 1), ("B", 0)):
        xs, ys, total = [], [], 0
        for label in data.labels:
            pool = data.by_label(label)
            mask = pool["chose_a"] == wanted if wanted == 1 else pool["chose_a"] == 0
            total += int(mask.sum())
            x, y = subsample([pool["rs_sel"][mask], pool["I0_sel"][mask]],
                             MAX_PER_CLASS, rng)
            xs.append(x)
            ys.append(y)
            del pool, mask
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        available += total
        plotted += x.size
        axes.scatter(x, y, s=2, alpha=0.12, linewidths=0, color=MODEL_COLORS[tag],
                     label=f"Model {tag} selected (n={total:,})")
        del xs, ys, x, y
    record = PlotRecord("I0_vs_rs_Model_Selected_by_AIC.png", available, plotted)
    records.append(record)
    axes.set_xlabel("rs_selected (1/ms)")
    axes.set_ylabel("I0_selected (a.u.)")
    axes.set_xlim(*limits[0])
    axes.set_ylim(*limits[1])
    axes.grid(alpha=0.25)
    axes.set_title("Model-selected parameter space, coloured by which model AIC chose\n"
                   f"{record.caption()}", fontsize=10)
    _legend(axes)
    _save(figure, folder / record.name, saved)
    progress(f"  {record.name}")

    # 5. four panels, one per sample, shared limits
    figure = _figure(figsize=(15, 12))
    panels = figure.subplots(2, 2)
    rng = np.random.default_rng(SEED)
    available = plotted = 0
    for axes, sample in zip(panels.ravel(), data.samples):
        sample_available, sample_plotted = _draw_by_material(
            axes, data, "rs_sel", "I0_sel", rng, sample=sample, limits=limits
        )
        available += sample_available
        plotted += sample_plotted
        axes.set_title(f"{sample} — {sample_available:,} voxels", fontsize=11)
        _legend(axes, loc="upper right")
    record = PlotRecord("I0_vs_rs_Model_Selected_by_sample.png", available, plotted)
    records.append(record)
    figure.suptitle("Model-selected I0 vs rs, per sample, identical axes\n"
                    f"{record.caption()}", fontsize=12)
    _save(figure, folder / record.name, saved)
    progress(f"  {record.name}")

    # 6. one figure per sample
    for sample in data.samples:
        figure = _figure(figsize=(10, 8.5))
        axes = figure.subplots()
        rng = np.random.default_rng(SEED)
        available, plotted = _draw_by_material(
            axes, data, "rs_sel", "I0_sel", rng, sample=sample, limits=limits
        )
        record = PlotRecord(f"I0_vs_rs_selected_{sample}.png", available, plotted)
        records.append(record)
        axes.set_xlabel("rs_selected (1/ms)")
        axes.set_ylabel("I0_selected (a.u.)")
        axes.set_title(f"{sample} — model-selected parameter space\n{record.caption()}",
                       fontsize=10)
        _legend(axes)
        _save(figure, folder / record.name, saved)
        progress(f"  {record.name}")

    # 7. three panels side by side, the same voxels in each
    figure = _figure(figsize=(19, 6.5))
    panels = figure.subplots(1, 3)
    available = plotted = 0
    for axes, (rs_key, i0_key, title) in zip(panels, (
        ("rs_A", "I0_A", "Model A (B fixed at 0)"),
        ("rs_B", "I0_B", "Model B (B free)"),
        ("rs_sel", "I0_sel", "Model-selected (AIC)"),
    )):
        # Same seed for each panel, so the same voxels are drawn in all three
        # and any change in structure is the representation, not the sample.
        rng = np.random.default_rng(SEED)
        panel_available, panel_plotted = _draw_by_material(
            axes, data, rs_key, i0_key, rng, limits=limits
        )
        available = panel_available
        plotted = panel_plotted
        axes.set_title(title, fontsize=11)
    _legend(panels[-1])
    record = PlotRecord("I0_vs_rs_Model_Comparison.png", available, plotted)
    records.append(record)
    figure.suptitle("Does allowing a baseline change the clustering? Identical voxels in all "
                    f"three panels.\n{record.caption()}", fontsize=11)
    _save(figure, folder / record.name, saved)
    progress(f"  {record.name}")

    # 9. B in the selected representation
    for rs_key, name, xlabel in (
        ("rs_sel", "rs_vs_B_Model_Selected.png", "rs_selected (1/ms)"),
        ("I0_sel", "I0_vs_B_Model_Selected.png", "I0_selected (a.u.)"),
    ):
        figure = _figure(figsize=(10, 8.5))
        axes = figure.subplots()
        rng = np.random.default_rng(SEED)
        available = plotted = 0
        for label in data.labels:
            pool = data.by_label(label)
            available += pool[rs_key].size
            x, y = subsample([pool[rs_key], pool["B_sel"]], MAX_PER_CLASS, rng)
            plotted += x.size
            axes.scatter(x, y, s=2, alpha=0.12, linewidths=0,
                         color=LABEL_COLORS[label], label=f"{label} (n={pool[rs_key].size:,})")
            del pool
        zero_fraction = _zero_fraction(data)
        record = PlotRecord(name, available, plotted)
        records.append(record)
        axes.set_xlabel(xlabel)
        axes.set_ylabel("B_selected (a.u.)")
        axes.grid(alpha=0.25)
        axes.set_title(f"{xlabel.split(' ')[0]} vs B in the model-selected representation\n"
                       f"B is exactly 0 for the {100 * zero_fraction:.1f}% of voxels where "
                       f"Model A won\n{record.caption()}", fontsize=10)
        _legend(axes)
        _save(figure, folder / name, saved)
        progress(f"  {name}")

    # 10. selection fractions
    records += _selection_figures(data, folder, saved, progress)
    return records


def _zero_fraction(data: SelectedData) -> float:
    """Fraction of voxels whose selected baseline is exactly zero."""
    zeros = total = 0
    for values in data.groups.values():
        zeros += int((values["B_sel"] == 0).sum())
        total += values["B_sel"].size
    return zeros / total if total else 0.0


def _selection_figures(data, folder, saved, progress) -> list[PlotRecord]:
    """Stacked bars of model choice, by material and by sample."""
    records = []
    for tag, keys, name, xlabel in (
        ("material", data.labels, "Model_Selection_By_Material.png", "material / concentration"),
        ("sample", data.samples, "Model_Selection_By_Sample.png", "sample"),
    ):
        figure = _figure(figsize=(11, 5.5))
        axes = figure.subplots()
        a_pct, b_pct, counts = [], [], []
        for key in keys:
            if tag == "material":
                pool = data.by_label(key)
                chose_a = pool["chose_a"]
            else:
                chose_a = np.concatenate([v["chose_a"] for (s, _), v in data.groups.items()
                                          if s == key])
            counts.append(chose_a.size)
            a_pct.append(100 * float((chose_a == 1).mean()))
            b_pct.append(100 * float((chose_a == 0).mean()))
            del chose_a
        positions = np.arange(len(keys))
        axes.bar(positions, a_pct, 0.62, label="Model A selected", color=MODEL_COLORS["A"])
        axes.bar(positions, b_pct, 0.62, bottom=a_pct, label="Model B selected",
                 color=MODEL_COLORS["B"])
        for position, value in zip(positions, a_pct):
            axes.text(position, min(value + 2, 95), f"{value:.1f}%", ha="center", fontsize=8)
        axes.set_xticks(positions)
        axes.set_xticklabels([f"{k}\n(n={c:,})" for k, c in zip(keys, counts)], fontsize=8)
        axes.set_xlabel(xlabel)
        axes.set_ylabel("% of voxels")
        axes.set_ylim(0, 105)
        axes.set_title(f"AIC model selection by {tag} — complete voxel population "
                       "(no subsampling)")
        axes.legend()
        axes.grid(alpha=0.25, axis="y")
        _save(figure, folder / name, saved)
        records.append(PlotRecord(name, sum(counts), 0))
        progress(f"  {name}")
    return records
