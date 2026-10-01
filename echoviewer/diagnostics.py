"""Diagnostic figures comparing Model A and Model B, and the class separation.

Reads the finished fits under ``i0_rs/Segmentations`` and writes PNGs under
``i0_rs/model_diagnostics``. Nothing here refits, and nothing here writes into
the fit results.

Two rules govern everything below:

* **Statistics use the complete valid voxel population.** Medians, IQRs,
  fractions and overlap measures are computed from every successful voxel.
* **Only the plotted points are subsampled**, with a fixed seed, and every
  such figure says so in its caption.

Memory: segmentations are read one at a time, only the needed columns, as
float32. What is retained across the run is seven 1D arrays per class - about
45 MB in total for 1.56M voxels - which is what allows exact medians and
overlap rather than approximations from a sample.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

#: Materials in the order they should appear on axes and in legends.
LABEL_ORDER = ("0", "0.05", "0.1", "0.2", "0.25", "0.3", "water", "tissue")

#: Numeric SPION concentrations, for the calibration plots.
CONCENTRATIONS = {"0": 0.0, "0.05": 0.05, "0.1": 0.1, "0.2": 0.2, "0.25": 0.25, "0.3": 0.3}

#: Colour per material: a sequential ramp for concentration, distinct hues for
#: the two biological/reference materials so they never read as a dose.
LABEL_COLORS = {
    "0": "#4a6fe3", "0.05": "#6ba3d6", "0.1": "#67c2a3",
    "0.2": "#e8b647", "0.25": "#e8783c", "0.3": "#c9314e",
    "water": "#8f8f8f", "tissue": "#8d5bd1",
}

#: Columns pulled from each voxel CSV. Anything not listed is never read.
CSV_COLUMNS = ("x", "y", "z", "I0_noB", "rs_noB", "I0_B", "rs_B", "B",
               "R2_noB", "R2_B", "AIC_noB", "AIC_B", "model_selected", "fit_success")

#: Maximum voxels drawn per class in any scatter plot.
MAX_PLOT_PER_CLASS = 150_000

#: Maximum points in a 3D scatter, which is far more expensive to render.
MAX_PLOT_3D = 100_000

SEED = 42

_SEG_NAME = re.compile(r"^Segmentation_(\d+)(?:_(.*))?$")

_SUBSAMPLE_NOTE = "Points subsampled for display; all statistics use every valid voxel."


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def iter_fit_dirs(fit_root: Path):
    """Yield ``(sample, segmentation_name, label, directory)`` for every fit."""
    for sample_dir in sorted(p for p in Path(fit_root).iterdir() if p.is_dir()):
        for seg_dir in sorted(p for p in sample_dir.iterdir() if p.is_dir()):
            match = _SEG_NAME.match(seg_dir.name)
            if match and (seg_dir / "voxel_fit_data.csv").is_file():
                yield sample_dir.name, seg_dir.name, (match.group(2) or ""), seg_dir


def read_voxels(directory: Path, columns=CSV_COLUMNS):
    """Read one segmentation's successful voxels as float32.

    Only the requested columns are parsed, and only successful fits are kept,
    so the frame stays proportional to the segmentation rather than the run.
    """
    import pandas as pd

    frame = pd.read_csv(
        directory / "voxel_fit_data.csv",
        usecols=list(columns),
        dtype={name: np.float32 for name in columns if name not in ("x", "y", "z")},
    )
    frame = frame[frame["fit_success"] > 0]
    return frame.reset_index(drop=True)


class ClassPools:
    """Per-material pools of the parameters needed for the summary figures.

    Holds one 1D array per parameter per material - a few tens of megabytes in
    total - so medians, IQRs and overlap can be computed exactly rather than
    estimated from the plotting subsample.
    """

    FIELDS = ("rs_noB", "rs_B", "I0_noB", "I0_B", "B", "delta_AIC", "model_selected")

    def __init__(self):
        self._chunks: dict[str, dict[str, list]] = {}

    def add(self, label: str, frame) -> None:
        """Append one segmentation's voxels to its material's pool."""
        store = self._chunks.setdefault(label, {name: [] for name in self.FIELDS})
        for name in self.FIELDS:
            if name == "delta_AIC":
                values = (frame["AIC_B"] - frame["AIC_noB"]).to_numpy(np.float32)
            else:
                values = frame[name].to_numpy(np.float32)
            store[name].append(values)

    def finish(self) -> dict[str, dict[str, np.ndarray]]:
        """Concatenate each material's chunks once, then drop the pieces."""
        pools = {}
        for label, store in self._chunks.items():
            pools[label] = {name: np.concatenate(parts) for name, parts in store.items()}
            store.clear()
        self._chunks.clear()
        return pools

    @property
    def labels(self) -> list[str]:
        return [name for name in LABEL_ORDER if name in self._chunks]


def ordered_labels(pools: dict) -> list[str]:
    """Materials present, in display order."""
    return [name for name in LABEL_ORDER if name in pools]


def subsample(arrays: list[np.ndarray], limit: int, rng) -> list[np.ndarray]:
    """Take the same random rows from several parallel arrays."""
    n = arrays[0].size
    if n <= limit:
        return arrays
    picked = rng.choice(n, size=limit, replace=False)
    return [array[picked] for array in arrays]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def iqr(values: np.ndarray) -> tuple[float, float]:
    """Interquartile range as ``(q1, q3)``."""
    return float(np.percentile(values, 25)), float(np.percentile(values, 75))


def bhattacharyya_overlap(first: np.ndarray, second: np.ndarray) -> float:
    """Overlap of two point clouds, via a Gaussian approximation.

    The Bhattacharyya coefficient ``exp(-D_B)`` runs from 1 (identical) to 0
    (disjoint), and works in any number of dimensions, which is what lets the
    same number compare ``rs`` alone against ``rs + I0`` and ``rs + I0 + B``.

    It assumes each class is roughly Gaussian in the feature space, so treat it
    as an indication of separability rather than an exact overlap. Distributions
    here are unimodal per material, which is the case it handles well.

    Args:
        first: ``(n_samples, n_features)``.
        second: ``(m_samples, n_features)``.

    Returns:
        Overlap in ``[0, 1]``; lower means better separated.
    """
    first = np.atleast_2d(first.T).T if first.ndim == 1 else first
    second = np.atleast_2d(second.T).T if second.ndim == 1 else second
    if first.shape[0] < 2 or second.shape[0] < 2:
        return float("nan")

    mean1, mean2 = first.mean(axis=0), second.mean(axis=0)
    cov1 = np.cov(first, rowvar=False).reshape(first.shape[1], -1)
    cov2 = np.cov(second, rowvar=False).reshape(second.shape[1], -1)
    pooled = 0.5 * (cov1 + cov2)

    # Ridge term: a class that is near-degenerate in one feature would give a
    # singular covariance and an infinite distance.
    ridge = 1e-12 * np.eye(pooled.shape[0]) * max(np.trace(pooled), 1e-12)
    pooled = pooled + ridge
    try:
        difference = (mean1 - mean2)[:, None]
        term = 0.125 * float(difference.T @ np.linalg.solve(pooled, difference))
        sign1, logdet1 = np.linalg.slogdet(cov1 + ridge)
        sign2, logdet2 = np.linalg.slogdet(cov2 + ridge)
        signp, logdetp = np.linalg.slogdet(pooled)
        if min(sign1, sign2, signp) <= 0:
            return float("nan")
        distance = term + 0.5 * (logdetp - 0.5 * (logdet1 + logdet2))
    except np.linalg.LinAlgError:
        return float("nan")
    return float(np.exp(-max(distance, 0.0)))


def overlap_1d(first: np.ndarray, second: np.ndarray, bins: int = 200) -> float:
    """Histogram overlap of two 1D samples, in ``[0, 1]``.

    Distribution-free, unlike the Bhattacharyya form, so it is used for the
    single-feature comparisons where no Gaussian assumption is needed.
    """
    low = min(first.min(), second.min())
    high = max(first.max(), second.max())
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return float("nan")
    edges = np.linspace(low, high, bins + 1)
    hist1, _ = np.histogram(first, bins=edges, density=True)
    hist2, _ = np.histogram(second, bins=edges, density=True)
    width = edges[1] - edges[0]
    return float(np.minimum(hist1, hist2).sum() * width)


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------


def _figure(*args, **kwargs):
    """A figure on the Agg canvas: headless, and independent of any GUI backend."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(*args, **kwargs)
    FigureCanvasAgg(figure)
    return figure


def _save(figure, path: Path, counter: list) -> None:
    """Write a PNG and release the figure immediately."""
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=110, bbox_inches="tight", facecolor="white")
    counter.append(path)
    # Figures are not kept open: hundreds of live figures is how this kind of
    # script runs a machine out of memory.
    figure.clf()
    del figure


def _violin(axes, pools, field, labels, ylabel, title, log=False):
    """Voxel-level distributions per material, median and IQR marked."""
    data = [pools[label][field] for label in labels]
    parts = axes.violinplot(data, showextrema=False, widths=0.85)
    for body, label in zip(parts["bodies"], labels):
        body.set_facecolor(LABEL_COLORS[label])
        body.set_alpha(0.65)
    for position, values in enumerate(data, start=1):
        q1, q3 = iqr(values)
        axes.vlines(position, q1, q3, color="black", lw=4, zorder=3)
        axes.plot(position, np.median(values), "o", color="white",
                  markeredgecolor="black", markersize=5, zorder=4)
    axes.set_xticks(range(1, len(labels) + 1))
    axes.set_xticklabels(labels, rotation=30, ha="right")
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    if log:
        axes.set_yscale("log")
    axes.grid(alpha=0.25, axis="y")


def _scatter_by_class(axes, pools, x_field, y_field, labels, rng,
                      xlabel, ylabel, title, limit=MAX_PLOT_PER_CLASS):
    """One scatter per material, drawn from a capped random subsample."""
    total = 0
    for label in labels:
        x, y = subsample(
            [pools[label][x_field], pools[label][y_field]], limit, rng
        )
        total += x.size
        axes.scatter(x, y, s=2, alpha=0.12, linewidths=0,
                     color=LABEL_COLORS[label], label=f"{label} (n={pools[label][x_field].size:,})")
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    axes.grid(alpha=0.25)
    legend = axes.legend(markerscale=6, fontsize=8, framealpha=0.9)
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
    return total


def _plane_views(volume: np.ndarray, mask: np.ndarray):
    """Indices of the busiest axial, coronal and sagittal slices of a mask."""
    return (
        int(np.argmax(mask.sum(axis=(1, 2)))),
        int(np.argmax(mask.sum(axis=(0, 2)))),
        int(np.argmax(mask.sum(axis=(0, 1)))),
    )


def _show_plane(axes, plane, title, cmap="viridis", vlim=None, symmetric=False):
    """Draw one masked parameter plane with a colour bar."""
    data = np.ma.masked_where(plane == 0, plane)
    kwargs = {"cmap": cmap, "interpolation": "nearest"}
    if vlim is not None:
        kwargs["vmin"], kwargs["vmax"] = vlim
    elif symmetric and data.count():
        limit = float(np.nanpercentile(np.abs(data.compressed()), 99)) or 1.0
        kwargs["vmin"], kwargs["vmax"] = -limit, limit
    elif data.count():
        kwargs["vmin"] = float(np.nanpercentile(data.compressed(), 1))
        kwargs["vmax"] = float(np.nanpercentile(data.compressed(), 99))
    image = axes.imshow(data, **kwargs)
    axes.set_title(title, fontsize=9)
    axes.set_axis_off()
    return image


# ---------------------------------------------------------------------------
# Model A vs Model B
# ---------------------------------------------------------------------------


def figures_model_comparison(pools, per_segment, out, rng, saved):
    """AIC, model preference, and how much rs and I0 move when B is added."""
    labels = ordered_labels(pools)
    folder = out / "model_comparison"

    all_aic_delta = np.concatenate([pools[l]["delta_AIC"] for l in labels])

    for field, name, title in (
        ("AIC_noB", "AIC_noB_distribution.png", "AIC, Model A"),
        ("AIC_B", "AIC_B_distribution.png", "AIC, Model B"),
    ):
        figure = _figure(figsize=(9, 5))
        axes = figure.subplots()
        for label in labels:
            values = per_segment["aic"][label][field]
            axes.hist(values, bins=200, histtype="step", lw=1.4,
                      color=LABEL_COLORS[label], label=f"{label} (n={values.size:,})",
                      density=True)
        axes.set_xlabel(field)
        axes.set_ylabel("density")
        axes.set_title(f"{title} — all valid voxels")
        axes.legend(fontsize=8)
        axes.grid(alpha=0.25)
        _save(figure, folder / name, saved)

    figure = _figure(figsize=(11, 5))
    left, right = figure.subplots(1, 2)
    for label in labels:
        values = pools[label]["delta_AIC"]
        left.hist(np.clip(values, -300, 300), bins=200, histtype="step", lw=1.4,
                  color=LABEL_COLORS[label], density=True, label=label)
    left.axvline(0, color="black", lw=1.2, ls="--")
    left.set_xlabel("delta_AIC = AIC_B − AIC_noB   (<0 favours Model B)")
    left.set_ylabel("density")
    left.set_title("delta_AIC by material (clipped to ±300 for display)")
    left.legend(fontsize=8)
    left.grid(alpha=0.25)

    fraction_b = float((all_aic_delta < 0).mean())
    right.bar(["Model A", "Model B"], [100 * (1 - fraction_b), 100 * fraction_b],
              color=["#4a6fe3", "#c9314e"])
    for index, value in enumerate([100 * (1 - fraction_b), 100 * fraction_b]):
        right.text(index, value + 1, f"{value:.1f}%", ha="center", fontweight="bold")
    right.set_ylabel("% of voxels preferred (AIC)")
    right.set_title(f"Model preference, all {all_aic_delta.size:,} voxels")
    right.set_ylim(0, 105)
    right.grid(alpha=0.25, axis="y")
    _save(figure, folder / "delta_AIC_distribution.png", saved)

    # Preference per material
    figure = _figure(figsize=(10, 5))
    axes = figure.subplots()
    a_pct = [100 * float((pools[l]["model_selected"] == 1).mean()) for l in labels]
    b_pct = [100 * float((pools[l]["model_selected"] == 2).mean()) for l in labels]
    positions = np.arange(len(labels))
    axes.bar(positions, a_pct, 0.62, label="Model A (noB)", color="#4a6fe3")
    axes.bar(positions, b_pct, 0.62, bottom=a_pct, label="Model B", color="#c9314e")
    for position, value in zip(positions, a_pct):
        axes.text(position, min(value + 2, 96), f"{value:.0f}%", ha="center", fontsize=8)
    axes.set_xticks(positions)
    axes.set_xticklabels([f"{l}\n(n={pools[l]['rs_B'].size:,})" for l in labels], fontsize=8)
    axes.set_ylabel("% of voxels")
    axes.set_title("Model preference by material (AIC) — complete voxel population")
    axes.legend()
    axes.grid(alpha=0.25, axis="y")
    _save(figure, folder / "model_preference_by_concentration.png", saved)

    figure = _figure(figsize=(7, 5))
    axes = figure.subplots()
    axes.bar(["Model A", "Model B"], [100 * (1 - fraction_b), 100 * fraction_b],
             color=["#4a6fe3", "#c9314e"])
    axes.set_ylabel("% of voxels")
    axes.set_title(f"Overall model preference ({all_aic_delta.size:,} voxels)")
    axes.grid(alpha=0.25, axis="y")
    _save(figure, folder / "model_preference_histogram.png", saved)

    # rs and I0: does adding B move them?
    for x_field, y_field, name, unit in (
        ("rs_noB", "rs_B", "rs_noB_vs_rs_B.png", "1/ms"),
        ("I0_noB", "I0_B", "I0_noB_vs_I0_B.png", "a.u."),
    ):
        figure = _figure(figsize=(8, 8))
        axes = figure.subplots()
        _scatter_by_class(axes, pools, x_field, y_field, labels, rng,
                          f"{x_field} ({unit})", f"{y_field} ({unit})",
                          f"{y_field} vs {x_field} — does adding B change it?")
        low = min(axes.get_xlim()[0], axes.get_ylim()[0])
        high = max(axes.get_xlim()[1], axes.get_ylim()[1])
        axes.plot([low, high], [low, high], "k--", lw=1.2, label="y = x")
        axes.set_xlim(low, high)
        axes.set_ylim(low, high)
        axes.text(0.02, 0.98, _SUBSAMPLE_NOTE, transform=axes.transAxes,
                  va="top", fontsize=7, style="italic")
        _save(figure, folder / name, saved)

    for a_field, b_field, name, unit in (
        ("rs_noB", "rs_B", "delta_rs_distribution.png", "1/ms"),
        ("I0_noB", "I0_B", "delta_I0_distribution.png", "a.u."),
    ):
        figure = _figure(figsize=(9, 5))
        axes = figure.subplots()
        for label in labels:
            delta = pools[label][b_field] - pools[label][a_field]
            axes.hist(delta, bins=200, histtype="step", lw=1.4, density=True,
                      color=LABEL_COLORS[label],
                      label=f"{label}: median {np.median(delta):+.4g}")
        axes.axvline(0, color="black", ls="--", lw=1.2)
        axes.set_xlabel(f"{b_field} − {a_field} ({unit})")
        axes.set_ylabel("density")
        axes.set_title(f"Change in {a_field.split('_')[0]} when B is introduced "
                       "— complete voxel population")
        axes.legend(fontsize=8)
        axes.grid(alpha=0.25)
        _save(figure, folder / name, saved)


def figures_spatial_model_comparison(seg_dir, seg_name, label, out, saved):
    """Axial/coronal/sagittal maps of both models at identical slices."""
    folder = out / "model_comparison"
    success = np.load(seg_dir / "fit_success_mask.npy") > 0
    planes = _plane_views(success, success)
    views = ("axial", "coronal", "sagittal")

    maps = {name: np.load(seg_dir / f"{name}.npy")
            for name in ("rs_map_noB", "rs_map_B", "I0_map_noB", "I0_map_B")}
    delta_rs = np.where(success, maps["rs_map_B"] - maps["rs_map_noB"], 0.0)
    delta_i0 = np.where(success, maps["I0_map_B"] - maps["I0_map_noB"], 0.0)

    def cut(volume, view, index):
        if view == "axial":
            return volume[index]
        if view == "coronal":
            return np.flipud(volume[:, index, :])
        return np.flipud(volume[:, :, index])

    for view, index in zip(views, planes):
        for source, name, cmap, symmetric in (
            (maps["rs_map_noB"], "rs_noB", "viridis", False),
            (maps["rs_map_B"], "rs_B", "viridis", False),
            (delta_rs, "delta_rs", "coolwarm", True),
            (maps["I0_map_noB"], "I0_noB", "magma", False),
            (maps["I0_map_B"], "I0_B", "magma", False),
            (delta_i0, "delta_I0", "coolwarm", True),
        ):
            figure = _figure(figsize=(5.5, 6))
            axes = figure.subplots()
            image = _show_plane(axes, cut(source, view, index),
                                f"{name} — {view} slice {index}\n{seg_name} ({label})",
                                cmap=cmap, symmetric=symmetric)
            figure.colorbar(image, ax=axes, fraction=0.045)
            _save(figure, folder / f"{name}_{view}.png", saved)
    del maps, delta_rs, delta_i0, success


# ---------------------------------------------------------------------------
# Distributions by material
# ---------------------------------------------------------------------------


def figures_distributions(pools, out, saved):
    """Voxel-level distributions of every parameter, per material."""
    labels = ordered_labels(pools)
    folder = out / "concentration_distributions"

    figure = _figure(figsize=(9, 5))
    axes = figure.subplots()
    every_b = np.concatenate([pools[l]["B"] for l in labels])
    axes.hist(every_b, bins=250, color="#8d5bd1", alpha=0.85)
    axes.axvline(float(np.median(every_b)), color="black", ls="--",
                 label=f"median {np.median(every_b):.1f}")
    axes.set_xlabel("B (a.u.)")
    axes.set_ylabel("voxels")
    axes.set_title(f"Baseline B, all {every_b.size:,} valid voxels "
                   f"({100 * float((every_b > 1).mean()):.1f}% above 1)")
    axes.legend()
    axes.grid(alpha=0.25)
    _save(figure, folder / "B_distribution_all.png", saved)
    del every_b

    for field, name, ylabel, title, log in (
        ("B", "B_by_concentration.png", "B (a.u.)", "Baseline B by material", False),
        ("B", "B_vs_concentration.png", "B (a.u.)", "B vs material", False),
        ("rs_B", "rs_vs_concentration.png", "rs_B (1/ms)", "Model B decay rate by material", False),
        ("rs_noB", "rs_noB_vs_concentration.png", "rs_noB (1/ms)", "Model A decay rate by material", False),
        ("I0_B", "I0_B_vs_concentration.png", "I0_B (a.u.)", "Model B amplitude by material", True),
        ("I0_noB", "I0_noB_vs_concentration.png", "I0_noB (a.u.)", "Model A amplitude by material", True),
    ):
        figure = _figure(figsize=(10, 5.5))
        axes = figure.subplots()
        _violin(axes, pools, field, labels, ylabel,
                f"{title} — every valid voxel (violin), median and IQR marked", log=log)
        _save(figure, folder / name, saved)


def figures_parameter_space(pools, out, rng, saved):
    """The scatter plots that decide whether the classes are separable."""
    labels = ordered_labels(pools)
    folder = out / "parameter_relationships"

    for x_field, y_field, name, title, xl, yl in (
        ("rs_noB", "I0_noB", "rs_I0_ModelA_by_concentration.png",
         "Model A parameter space", "rs_noB (1/ms)", "I0_noB (a.u.)"),
        ("rs_B", "I0_B", "rs_I0_ModelB_by_concentration.png",
         "Model B parameter space", "rs_B (1/ms)", "I0_B (a.u.)"),
        ("rs_B", "B", "rs_B_vs_B_by_concentration.png",
         "rs_B vs B", "rs_B (1/ms)", "B (a.u.)"),
        ("I0_B", "B", "I0_B_vs_B_by_concentration.png",
         "I0_B vs B", "I0_B (a.u.)", "B (a.u.)"),
    ):
        figure = _figure(figsize=(9, 8))
        axes = figure.subplots()
        _scatter_by_class(axes, pools, x_field, y_field, labels, rng, xl, yl,
                          f"{title} — one point per voxel, coloured by material")
        axes.text(0.02, 0.98, _SUBSAMPLE_NOTE, transform=axes.transAxes,
                  va="top", fontsize=7, style="italic")
        _save(figure, folder / name, saved)

    # 3D feature space
    figure = _figure(figsize=(10, 9))
    axes = figure.add_subplot(projection="3d")
    per_class = max(MAX_PLOT_3D // max(len(labels), 1), 1000)
    for label in labels:
        x, y, z = subsample(
            [pools[label]["rs_B"], pools[label]["I0_B"], pools[label]["B"]],
            per_class, rng,
        )
        axes.scatter(x, y, z, s=2, alpha=0.15, linewidths=0,
                     color=LABEL_COLORS[label], label=label)
    axes.set_xlabel("rs_B (1/ms)")
    axes.set_ylabel("I0_B (a.u.)")
    axes.set_zlabel("B (a.u.)")
    axes.set_title(f"(rs_B, I0_B, B) feature space — {_SUBSAMPLE_NOTE}", fontsize=9)
    axes.legend(markerscale=6, fontsize=8)
    _save(figure, folder / "rs_I0_B_3D_by_concentration.png", saved)


def figure_separation_matrix(pools, out, saved):
    """Pairwise class overlap under three feature sets."""
    labels = ordered_labels(pools)
    folder = out / "parameter_relationships"
    feature_sets = (
        ("rs_B", ("rs_B",)),
        ("rs_B + I0_B", ("rs_B", "I0_B")),
        ("rs_B + I0_B + B", ("rs_B", "I0_B", "B")),
    )

    figure = _figure(figsize=(17, 5.4))
    panels = figure.subplots(1, 3)
    matrices = {}
    for axes, (title, fields) in zip(panels, feature_sets):
        matrix = np.full((len(labels), len(labels)), np.nan)
        for i, first in enumerate(labels):
            for j, second in enumerate(labels):
                if i == j:
                    matrix[i, j] = 1.0
                    continue
                # One measure for all three panels. Mixing a histogram overlap
                # into the 1D panel and a Bhattacharyya coefficient into the
                # others would put the columns on different scales, and the
                # whole point is to compare them against each other.
                a = np.column_stack([pools[first][f] for f in fields])
                b = np.column_stack([pools[second][f] for f in fields])
                matrix[i, j] = bhattacharyya_overlap(a, b)
        matrices[title] = matrix
        image = axes.imshow(matrix, cmap="RdYlGn_r", vmin=0, vmax=1)
        axes.set_xticks(range(len(labels)))
        axes.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        axes.set_yticks(range(len(labels)))
        axes.set_yticklabels(labels, fontsize=8)
        axes.set_title(f"Overlap using {title}", fontsize=10)
        for i in range(len(labels)):
            for j in range(len(labels)):
                if np.isfinite(matrix[i, j]):
                    axes.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                              fontsize=7,
                              color="white" if matrix[i, j] > 0.6 else "black")
        figure.colorbar(image, ax=axes, fraction=0.046)
    figure.suptitle("Pairwise class overlap (0 = separated, 1 = identical). "
                    "Bhattacharyya coefficient with a Gaussian approximation "
                    "throughout, so the three panels are directly comparable. "
                    "Complete voxel populations.", fontsize=9)
    _save(figure, folder / "class_separation_matrix.png", saved)
    return matrices, labels


# ---------------------------------------------------------------------------
# Calibration, tissue vs 0.1
# ---------------------------------------------------------------------------


def figures_calibration(sample_pools, out, rng, saved):
    """Known concentrations only, and whether the phantoms agree."""
    folder = out / "calibration"
    calib_samples = [s for s in ("Syringes", "3D") if s in sample_pools]

    # Pooled calibration curves
    for field, name, ylabel in (
        ("rs_B", "calibration_rs.png", "rs_B (1/ms)"),
        ("I0_B", "calibration_I0.png", "I0_B (a.u.)"),
        ("B", "calibration_B.png", "B (a.u.)"),
    ):
        figure = _figure(figsize=(10, 5.5))
        axes = figure.subplots()
        offsets = {"Syringes": -0.008, "3D": 0.008, "Pill1": -0.02, "Pill2": 0.02}
        for sample in sorted(sample_pools):
            points = []
            for label, values in sample_pools[sample].items():
                if label not in CONCENTRATIONS:
                    continue
                data = values[field]
                q1, q3 = iqr(data)
                x = CONCENTRATIONS[label] + offsets.get(sample, 0.0)
                axes.vlines(x, q1, q3, lw=3, alpha=0.6,
                            color=LABEL_COLORS.get(label, "#666"))
                points.append((x, float(np.median(data)), data.size))
            if points:
                points.sort()
                axes.plot([p[0] for p in points], [p[1] for p in points], "o-",
                          label=f"{sample}", lw=1.4, markersize=6)
        axes.set_xlabel("known SPION concentration (mg/mL)")
        axes.set_ylabel(ylabel)
        axes.set_title(f"Calibration: {field} vs concentration — median and IQR "
                       "over every valid voxel\n(lines connect medians; descriptive only, not a fit)")
        axes.legend(fontsize=8)
        axes.grid(alpha=0.25)
        _save(figure, folder / name, saved)

    # Joint rs/I0 for calibration samples
    figure = _figure(figsize=(9, 8))
    axes = figure.subplots()
    for sample in calib_samples:
        for label, values in sample_pools[sample].items():
            if label not in CONCENTRATIONS:
                continue
            x, y = subsample([values["rs_B"], values["I0_B"]], 40_000, rng)
            axes.scatter(x, y, s=2, alpha=0.12, linewidths=0,
                         color=LABEL_COLORS[label],
                         marker="o" if sample == "Syringes" else "^",
                         label=f"{sample} {label}")
    axes.set_xlabel("rs_B (1/ms)")
    axes.set_ylabel("I0_B (a.u.)")
    axes.set_title(f"Calibration joint distribution (circles Syringes, triangles 3D)\n{_SUBSAMPLE_NOTE}")
    legend = axes.legend(markerscale=6, fontsize=7, ncol=2)
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
    axes.grid(alpha=0.25)
    _save(figure, folder / "calibration_rs_I0.png", saved)

    # Phantom agreement at matched concentrations
    shared = sorted(
        {l for s in sample_pools for l in sample_pools[s] if l in CONCENTRATIONS},
        key=lambda v: CONCENTRATIONS[v],
    )
    for field, name, ylabel in (
        ("rs_B", "syringes_vs_3D_rs.png", "rs_B (1/ms)"),
        ("I0_B", "syringes_vs_3D_I0.png", "I0_B (a.u.)"),
        ("B", "syringes_vs_3D_B.png", "B (a.u.)"),
    ):
        figure = _figure(figsize=(12, 5.5))
        axes = figure.subplots()
        width = 0.2
        markers = {"Syringes": "#4a6fe3", "3D": "#c9314e", "Pill1": "#e8b647", "Pill2": "#67c2a3"}
        for offset, (sample, colour) in enumerate(markers.items()):
            if sample not in sample_pools:
                continue
            xs, medians, lows, highs = [], [], [], []
            for position, label in enumerate(shared):
                pool = sample_pools[sample].get(label)
                if pool is None:
                    continue
                q1, q3 = iqr(pool[field])
                xs.append(position + (offset - 1.5) * width)
                medians.append(float(np.median(pool[field])))
                lows.append(q1)
                highs.append(q3)
            if xs:
                axes.errorbar(xs, medians,
                              yerr=[np.array(medians) - lows, np.array(highs) - medians],
                              fmt="o", color=colour, capsize=3, label=sample, markersize=6)
        axes.set_xticks(range(len(shared)))
        axes.set_xticklabels(shared)
        axes.set_xlabel("known SPION concentration (mg/mL)")
        axes.set_ylabel(ylabel)
        axes.set_title(f"{field} at matched concentrations across phantoms — "
                       "median with IQR, complete voxel populations")
        axes.legend()
        axes.grid(alpha=0.25)
        _save(figure, folder / name, saved)

    figure = _figure(figsize=(9, 8))
    axes = figure.subplots()
    style = {"Syringes": "o", "3D": "^", "Pill1": "s", "Pill2": "D"}
    for sample in sorted(sample_pools):
        for label, values in sample_pools[sample].items():
            if label not in CONCENTRATIONS:
                continue
            x, y = subsample([values["rs_B"], values["I0_B"]], 25_000, rng)
            axes.scatter(x, y, s=2, alpha=0.12, linewidths=0,
                         color=LABEL_COLORS[label], marker=style.get(sample, "o"))
    axes.set_xlabel("rs_B (1/ms)")
    axes.set_ylabel("I0_B (a.u.)")
    axes.set_title("All phantoms, known concentrations, in (rs_B, I0_B)\n"
                   "colour = concentration, marker = phantom. " + _SUBSAMPLE_NOTE)
    axes.grid(alpha=0.25)
    _save(figure, folder / "syringes_vs_3D_rs_I0.png", saved)


def figures_tissue_vs_spion(pools, out, rng, saved, target="0.1"):
    """The decisive comparison: tissue against ~0.1 mg/mL SPION."""
    folder = out / "tissue_vs_spion"
    if "tissue" not in pools or target not in pools:
        return {}
    tissue, spion = pools["tissue"], pools[target]

    stats = {}
    for field, name, xlabel in (
        ("rs_B", f"tissue_vs_{target}_rs.png", "rs_B (1/ms)"),
        ("I0_B", f"tissue_vs_{target}_I0.png", "I0_B (a.u.)"),
        ("B", f"tissue_vs_{target}_B.png", "B (a.u.)"),
    ):
        # Bhattacharyya for the headline number, so it is on the same scale
        # as the 2D and 3D figures below; the histogram overlap is reported
        # alongside because it makes no distributional assumption.
        overlap = bhattacharyya_overlap(tissue[field], spion[field])
        histogram = overlap_1d(tissue[field], spion[field])
        stats[field] = overlap
        stats[f"{field}_histogram"] = histogram
        figure = _figure(figsize=(9, 5))
        axes = figure.subplots()
        for values, label, colour in ((tissue[field], "tissue", LABEL_COLORS["tissue"]),
                                      (spion[field], f"{target} mg/mL", LABEL_COLORS[target])):
            axes.hist(values, bins=200, density=True, alpha=0.55, color=colour,
                      label=f"{label} (n={values.size:,}, median {np.median(values):.4g})")
        axes.set_xlabel(xlabel)
        axes.set_ylabel("density")
        axes.set_title(f"tissue vs {target} mg/mL SPION — {field}\n"
                       f"overlap: Bhattacharyya {overlap:.3f}, histogram {histogram:.3f} "
                       "(complete voxel populations)")
        axes.legend(fontsize=8)
        axes.grid(alpha=0.25)
        _save(figure, folder / name, saved)

    two = bhattacharyya_overlap(
        np.column_stack([tissue["rs_B"], tissue["I0_B"]]),
        np.column_stack([spion["rs_B"], spion["I0_B"]]),
    )
    three = bhattacharyya_overlap(
        np.column_stack([tissue["rs_B"], tissue["I0_B"], tissue["B"]]),
        np.column_stack([spion["rs_B"], spion["I0_B"], spion["B"]]),
    )
    stats["rs_B+I0_B"] = two
    stats["rs_B+I0_B+B"] = three

    figure = _figure(figsize=(9, 8))
    axes = figure.subplots()
    for values, label, colour in ((tissue, "tissue", LABEL_COLORS["tissue"]),
                                  (spion, f"{target} mg/mL", LABEL_COLORS[target])):
        x, y = subsample([values["rs_B"], values["I0_B"]], MAX_PLOT_PER_CLASS, rng)
        axes.scatter(x, y, s=3, alpha=0.15, linewidths=0, color=colour,
                     label=f"{label} (n={values['rs_B'].size:,})")
    axes.set_xlabel("rs_B (1/ms)")
    axes.set_ylabel("I0_B (a.u.)")
    axes.set_title(f"tissue vs {target} mg/mL in (rs_B, I0_B)\n"
                   f"overlap: rs alone {stats['rs_B']:.3f} → with I0 {two:.3f}. {_SUBSAMPLE_NOTE}")
    legend = axes.legend(markerscale=6)
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
    axes.grid(alpha=0.25)
    _save(figure, folder / f"tissue_vs_{target}_rs_I0.png", saved)

    figure = _figure(figsize=(10, 9))
    axes = figure.add_subplot(projection="3d")
    for values, label, colour in ((tissue, "tissue", LABEL_COLORS["tissue"]),
                                  (spion, f"{target} mg/mL", LABEL_COLORS[target])):
        x, y, z = subsample([values["rs_B"], values["I0_B"], values["B"]], 50_000, rng)
        axes.scatter(x, y, z, s=2, alpha=0.15, linewidths=0, color=colour, label=label)
    axes.set_xlabel("rs_B (1/ms)")
    axes.set_ylabel("I0_B (a.u.)")
    axes.set_zlabel("B (a.u.)")
    axes.set_title(f"tissue vs {target} mg/mL in (rs_B, I0_B, B)\n"
                   f"overlap {three:.3f} vs {two:.3f} in 2D. {_SUBSAMPLE_NOTE}", fontsize=9)
    axes.legend(markerscale=6)
    _save(figure, folder / f"tissue_vs_{target}_rs_I0_B_3D.png", saved)
    return stats


# ---------------------------------------------------------------------------
# Voxel decay curves and residuals
# ---------------------------------------------------------------------------


def load_voxel_curves(seg_source_dir: Path, coords):
    """Measured intensity at every echo time for a handful of voxels.

    Reads the exported ``.npz`` volumes one echo at a time and keeps only the
    requested voxels, so this never holds more than one volume at once.
    """
    folders = []
    for entry in sorted(seg_source_dir.iterdir()):
        if entry.is_dir():
            number = re.search(r"[-+]?\d*\.?\d+", entry.name)
            if number and any(entry.glob("*.npz")):
                folders.append((float(number.group()), entry))
    folders.sort(key=lambda pair: pair[0])

    echo_times = np.array([value for value, _ in folders], dtype=float)
    curves = np.zeros((len(coords), len(folders)), dtype=np.float32)
    for column, (_, folder) in enumerate(folders):
        with np.load(sorted(folder.glob("*.npz"))[0]) as archive:
            volume = archive["volume"] if "volume" in archive else archive[archive.files[0]]
            for row, (z, y, x) in enumerate(coords):
                curves[row, column] = volume[z, y, x]
        del volume
    return echo_times, curves


def _draw_decay(axes, te, curve, row, title):
    """Measured points with both fitted models overlaid."""
    fit_a = row["I0_noB"] * np.exp(-row["rs_noB"] * te)
    fit_b = row["I0_B"] * np.exp(-row["rs_B"] * te) + row["B"]
    axes.plot(te, curve, "ko", markersize=4, label="measured")
    axes.plot(te, fit_a, "-", color="#4a6fe3", lw=1.6,
              label=f"A: I0={row['I0_noB']:.0f} rs={row['rs_noB']:.4f}\n"
                    f"    R2={row['R2_noB']:.4f} AIC={row['AIC_noB']:.1f}")
    axes.plot(te, fit_b, "-", color="#c9314e", lw=1.6,
              label=f"B: I0={row['I0_B']:.0f} rs={row['rs_B']:.4f} B={row['B']:.1f}\n"
                    f"    R2={row['R2_B']:.4f} AIC={row['AIC_B']:.1f}")
    axes.set_xlabel("Echo time (ms)")
    axes.set_ylabel("intensity (a.u.)")
    axes.set_title(title, fontsize=9)
    axes.legend(fontsize=7)
    axes.grid(alpha=0.25)


def figures_voxel_examples(fit_root, seg_root, examples, out, saved):
    """One decay curve per material, plus the three AIC cases, plus residuals."""
    folder = out / "voxel_examples"
    residual_rows = []

    for tag, (sample, seg_name, label, row) in examples.items():
        coords = [(int(row["z"]), int(row["y"]), int(row["x"]))]
        source = Path(seg_root) / sample / "volumes" / seg_name
        te, curves = load_voxel_curves(source, coords)
        curve = curves[0].astype(float)

        figure = _figure(figsize=(8, 5.5))
        axes = figure.subplots()
        _draw_decay(axes, te, curve, row,
                    f"{tag}: {sample} / {seg_name} ({label})\n"
                    f"voxel (x,y,z)=({int(row['x'])},{int(row['y'])},{int(row['z'])})  "
                    f"delta_AIC={row['AIC_B'] - row['AIC_noB']:+.1f}")
        _save(figure, folder / f"decay_{tag}.png", saved)
        residual_rows.append((tag, sample, seg_name, label, te, curve, row))
        del curves

    for model, name in (("A", "residual_ModelA_examples.png"),
                        ("B", "residual_ModelB_examples.png")):
        count = len(residual_rows)
        figure = _figure(figsize=(13, 3.1 * ((count + 2) // 3)))
        panels = figure.subplots((count + 2) // 3, 3, squeeze=False).ravel()
        for axes, (tag, sample, seg_name, label, te, curve, row) in zip(panels, residual_rows):
            if model == "A":
                fitted = row["I0_noB"] * np.exp(-row["rs_noB"] * te)
                colour = "#4a6fe3"
            else:
                fitted = row["I0_B"] * np.exp(-row["rs_B"] * te) + row["B"]
                colour = "#c9314e"
            residual = curve - fitted
            axes.axhline(0, color="black", lw=1)
            axes.plot(te, residual, "o-", color=colour, markersize=3, lw=1)
            axes.set_title(f"{tag} ({label})\nmean residual {residual.mean():+.1f}", fontsize=8)
            axes.set_xlabel("TE (ms)")
            axes.set_ylabel("measured − fitted")
            axes.grid(alpha=0.25)
        for axes in panels[len(residual_rows):]:
            axes.set_axis_off()
        figure.suptitle(f"Model {model} residuals — a systematic offset means the model "
                        "is missing a baseline", fontsize=10)
        _save(figure, folder / name, saved)


# ---------------------------------------------------------------------------
# Per-segmentation overview
# ---------------------------------------------------------------------------


def figure_segment_overview(sample, seg_name, label, seg_dir, seg_source, frame,
                            out, rng, saved):
    """The first picture to open when inspecting one segmentation."""
    success = np.load(seg_dir / "fit_success_mask.npy") > 0
    axial, _, _ = _plane_views(success, success)

    maps = {name: np.load(seg_dir / f"{name}.npy")
            for name in ("rs_map_B", "I0_map_B", "B_map", "R2_B_map")}
    delta_aic = np.zeros(success.shape, dtype=np.float32)
    delta_aic[frame["z"].to_numpy(np.intp), frame["y"].to_numpy(np.intp),
              frame["x"].to_numpy(np.intp)] = (frame["AIC_B"] - frame["AIC_noB"]).to_numpy(np.float32)

    figure = _figure(figsize=(16, 8.5))
    grid = figure.add_gridspec(2, 4, hspace=0.28, wspace=0.22)
    for position, (name, source, cmap, symmetric) in enumerate((
        ("rs_B", maps["rs_map_B"], "viridis", False),
        ("I0_B", maps["I0_map_B"], "magma", False),
        ("B", maps["B_map"], "cividis", False),
        ("R2_B", maps["R2_B_map"], "plasma", False),
        ("delta_AIC (<0 favours B)", delta_aic, "coolwarm", True),
    )):
        axes = figure.add_subplot(grid[position // 4, position % 4])
        image = _show_plane(axes, source[axial], f"{name} — axial {axial}",
                            cmap=cmap, symmetric=symmetric)
        figure.colorbar(image, ax=axes, fraction=0.045)

    axes = figure.add_subplot(grid[1, 1])
    x, y = subsample([frame["rs_B"].to_numpy(np.float32),
                      frame["I0_B"].to_numpy(np.float32)], 60_000, rng)
    axes.scatter(x, y, s=2, alpha=0.12, linewidths=0, color=LABEL_COLORS.get(label, "#666"))
    axes.set_xlabel("rs_B (1/ms)")
    axes.set_ylabel("I0_B (a.u.)")
    axes.set_title(f"rs_B vs I0_B ({len(frame):,} voxels; {x.size:,} drawn)", fontsize=9)
    axes.grid(alpha=0.25)

    axes = figure.add_subplot(grid[1, 2:])
    best = frame.iloc[int(np.argmax(frame["R2_B"].to_numpy()))]
    te, curves = load_voxel_curves(seg_source, [(int(best["z"]), int(best["y"]), int(best["x"]))])
    _draw_decay(axes, te, curves[0].astype(float), best,
                f"representative voxel (x,y,z)="
                f"({int(best['x'])},{int(best['y'])},{int(best['z'])})")

    fraction_b = 100 * float((frame["model_selected"] == 2).mean())
    figure.suptitle(
        f"{sample} / {seg_name}   material: {label}   "
        f"{len(frame):,} voxels   Model B preferred in {fraction_b:.1f}%",
        fontsize=13, fontweight="600",
    )
    _save(figure, out / "summary" / "segmentations" / sample / f"{seg_name}_overview.png", saved)
    _save(_spatial_overview(sample, seg_name, label, maps, delta_aic, success),
          out / "summary" / f"spatial_overview_{sample}_{seg_name}.png", saved)
    del maps, delta_aic, success, curves


def _spatial_overview(sample, seg_name, label, maps, delta_aic, success):
    """Three planes of the Model B parameters, for a spatial sanity check."""
    axial, coronal, sagittal = _plane_views(success, success)
    figure = _figure(figsize=(15, 9))
    panels = figure.subplots(3, 5)
    sources = (("rs_B", maps["rs_map_B"], "viridis", False),
               ("I0_B", maps["I0_map_B"], "magma", False),
               ("B", maps["B_map"], "cividis", False),
               ("R2_B", maps["R2_B_map"], "plasma", False),
               ("delta_AIC", delta_aic, "coolwarm", True))
    cuts = (("axial", lambda v: v[axial]),
            ("coronal", lambda v: np.flipud(v[:, coronal, :])),
            ("sagittal", lambda v: np.flipud(v[:, :, sagittal])))
    for row, (view, cut) in enumerate(cuts):
        for column, (name, source, cmap, symmetric) in enumerate(sources):
            image = _show_plane(panels[row][column], cut(source),
                                f"{name} — {view}", cmap=cmap, symmetric=symmetric)
            figure.colorbar(image, ax=panels[row][column], fraction=0.045)
    figure.suptitle(f"Spatial sanity check — {sample} / {seg_name} ({label})",
                    fontsize=13, fontweight="600")
    return figure


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run(fit_root, seg_root, out_root, progress=print):
    """Generate every diagnostic figure and the summary workbook.

    Segmentations are read one at a time; only the columns needed are parsed,
    and every figure is closed as soon as it is written.
    """
    import pandas as pd

    fit_root, seg_root = Path(fit_root), Path(seg_root)
    out = Path(out_root)
    for name in ("model_comparison", "concentration_distributions", "parameter_relationships",
                 "calibration", "tissue_vs_spion", "voxel_examples", "summary"):
        (out / name).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    saved: list[Path] = []
    pools_builder = ClassPools()
    sample_builders: dict[str, ClassPools] = {}
    per_segment_rows = []
    aic_pool: dict[str, dict[str, list]] = {}
    examples: dict[str, tuple] = {}
    best_equal = (np.inf, None)

    entries = list(iter_fit_dirs(fit_root))
    progress(f"found {len(entries)} segmentations")

    for index, (sample, seg_name, label, seg_dir) in enumerate(entries, start=1):
        frame = read_voxels(seg_dir)
        progress(f"  [{index}/{len(entries)}] {sample}/{seg_name} ({label}) "
                 f"{len(frame):,} voxels")

        delta_aic = (frame["AIC_B"] - frame["AIC_noB"]).to_numpy(np.float32)
        per_segment_rows.append({
            "sample": sample, "segmentation": seg_name, "material": label,
            "n_voxels": int(len(frame)),
            "fraction_ModelA": float((frame["model_selected"] == 1).mean()),
            "fraction_ModelB": float((frame["model_selected"] == 2).mean()),
            "median_delta_AIC": float(np.median(delta_aic)),
            "median_delta_rs": float(np.median(frame["rs_B"] - frame["rs_noB"])),
            "median_delta_I0": float(np.median(frame["I0_B"] - frame["I0_noB"])),
            "median_B": float(np.median(frame["B"])),
            "median_rs_B": float(np.median(frame["rs_B"])),
            "median_I0_B": float(np.median(frame["I0_B"])),
        })

        pools_builder.add(label, frame)
        sample_builders.setdefault(sample, ClassPools()).add(label, frame)
        store = aic_pool.setdefault(label, {"AIC_noB": [], "AIC_B": []})
        store["AIC_noB"].append(frame["AIC_noB"].to_numpy(np.float32))
        store["AIC_B"].append(frame["AIC_B"].to_numpy(np.float32))

        # One representative voxel per material, plus the three AIC cases.
        if label not in examples:
            examples[label] = (sample, seg_name, label,
                               frame.iloc[int(np.argmax(frame["R2_B"].to_numpy()))])
        strongest_a = int(np.argmax(delta_aic))
        strongest_b = int(np.argmin(delta_aic))
        if "ModelA_clearly_preferred" not in examples or \
                delta_aic[strongest_a] > examples["ModelA_clearly_preferred"][3]["AIC_B"] - \
                examples["ModelA_clearly_preferred"][3]["AIC_noB"]:
            examples["ModelA_clearly_preferred"] = (sample, seg_name, label, frame.iloc[strongest_a])
        if "ModelB_clearly_preferred" not in examples or \
                delta_aic[strongest_b] < examples["ModelB_clearly_preferred"][3]["AIC_B"] - \
                examples["ModelB_clearly_preferred"][3]["AIC_noB"]:
            examples["ModelB_clearly_preferred"] = (sample, seg_name, label, frame.iloc[strongest_b])
        tied = int(np.argmin(np.abs(delta_aic)))
        if abs(float(delta_aic[tied])) < best_equal[0]:
            best_equal = (abs(float(delta_aic[tied])), (sample, seg_name, label, frame.iloc[tied]))

        figure_segment_overview(sample, seg_name, label, seg_dir,
                                seg_root / sample / "volumes" / seg_name,
                                frame, out, rng, saved)
        del frame, delta_aic

    if best_equal[1] is not None:
        examples["models_equivalent"] = best_equal[1]

    pools = pools_builder.finish()
    sample_pools = {name: builder.finish() for name, builder in sample_builders.items()}
    aic = {label: {key: np.concatenate(parts) for key, parts in store.items()}
           for label, store in aic_pool.items()}
    aic_pool.clear()

    progress("figures: model comparison")
    figures_model_comparison(pools, {"aic": aic}, out, rng, saved)
    biggest = max(entries, key=lambda e: (e[3] / "voxel_fit_data.csv").stat().st_size)
    figures_spatial_model_comparison(biggest[3], biggest[1], biggest[2], out, saved)
    del aic

    progress("figures: distributions and parameter space")
    figures_distributions(pools, out, saved)
    figures_parameter_space(pools, out, rng, saved)
    matrices, matrix_labels = figure_separation_matrix(pools, out, saved)

    progress("figures: B spatial maps")
    figure = _figure(figsize=(15, 4.2))
    panels = figure.subplots(1, 4)
    for axes, (sample, seg_name, label, seg_dir) in zip(panels, entries[:4]):
        success = np.load(seg_dir / "fit_success_mask.npy") > 0
        b_map = np.load(seg_dir / "B_map.npy")
        axial = int(np.argmax(success.sum(axis=(1, 2))))
        image = _show_plane(axes, b_map[axial], f"{sample}\n{seg_name} ({label})", cmap="cividis")
        figure.colorbar(image, ax=axes, fraction=0.045)
        del success, b_map
    figure.suptitle("Baseline B in space — is it structured or uniform?", fontsize=12)
    _save(figure, out / "concentration_distributions" / "B_spatial_maps.png", saved)

    progress("figures: calibration")
    figures_calibration(sample_pools, out, rng, saved)

    progress("figures: tissue vs 0.1")
    tissue_stats = figures_tissue_vs_spion(pools, out, rng, saved)

    progress("figures: voxel decay curves")
    figures_voxel_examples(fit_root, seg_root, examples, out, saved)

    progress("workbook")
    workbook = out / "summary" / "diagnostic_summary.xlsx"
    labels = ordered_labels(pools)
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame(per_segment_rows).to_excel(writer, sheet_name="Model_Comparison", index=False)

        rows = []
        for label in labels:
            pool = pools[label]
            entry = {"material": label, "n_voxels": int(pool["rs_B"].size)}
            for field in ("rs_B", "I0_B", "B", "rs_noB", "I0_noB"):
                q1, q3 = iqr(pool[field])
                entry[f"median_{field}"] = float(np.median(pool[field]))
                entry[f"IQR_{field}"] = q3 - q1
                entry[f"q1_{field}"] = q1
                entry[f"q3_{field}"] = q3
            entry["fraction_ModelB"] = float((pool["model_selected"] == 2).mean())
            rows.append(entry)
        pd.DataFrame(rows).to_excel(writer, sheet_name="Concentration_Summary", index=False)

        pd.DataFrame([{"comparison": "tissue vs 0.1 mg/mL", "feature_set": key,
                       "overlap": value} for key, value in tissue_stats.items()]
                     ).to_excel(writer, sheet_name="Tissue_vs_0.1", index=False)

        calib_rows = []
        for sample in sorted(sample_pools):
            for label, pool in sample_pools[sample].items():
                if label not in CONCENTRATIONS:
                    continue
                q1, q3 = iqr(pool["rs_B"])
                i1, i3 = iqr(pool["I0_B"])
                b1, b3 = iqr(pool["B"])
                calib_rows.append({
                    "sample": sample, "concentration": CONCENTRATIONS[label],
                    "n_voxels": int(pool["rs_B"].size),
                    "median_rs_B": float(np.median(pool["rs_B"])), "IQR_rs_B": q3 - q1,
                    "median_I0_B": float(np.median(pool["I0_B"])), "IQR_I0_B": i3 - i1,
                    "median_B": float(np.median(pool["B"])), "IQR_B": b3 - b1,
                })
        pd.DataFrame(calib_rows).to_excel(writer, sheet_name="Calibration", index=False)

        overlap_rows = []
        for title, matrix in matrices.items():
            for i, first in enumerate(matrix_labels):
                for j, second in enumerate(matrix_labels):
                    if i < j:
                        overlap_rows.append({"feature_set": title, "class_a": first,
                                             "class_b": second, "overlap": matrix[i, j]})
        pd.DataFrame(overlap_rows).to_excel(writer, sheet_name="Class_Overlap", index=False)
    saved.append(workbook)

    return pools, sample_pools, per_segment_rows, matrices, matrix_labels, tissue_stats, saved
