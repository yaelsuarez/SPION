"""Voxel-wise exponential decay fitting, with and without a baseline offset.

Every voxel is fitted independently across echo times — nothing is averaged
before fitting, so the per-voxel distributions of ``I0``, ``rs`` and ``B``
survive for later classification.

Two models, both kept::

    Model A (noB):  I(TE) = I0 * exp(-rs * TE)
    Model B (B):    I(TE) = I0 * exp(-rs * TE) + B

Neither is discarded and neither is assumed correct. ``B`` is free, so Model B
can report a genuine noise floor if the data holds one.

**Model choice.** ``model_choice_map`` records which model *Akaike's
Information Criterion* prefers — ``1`` for Model A, ``2`` for Model B, ``0``
where nothing was fitted. AIC rather than R², because R² can only rise when a
third parameter is added and would pick Model B almost everywhere regardless of
whether the baseline is real::

    AIC = n * ln(RSS / n) + 2k

Ties go to Model A, the simpler one. This map is a convenience: every
underlying number is saved separately, so any other criterion can be applied
afterwards.

**Parameter bounds.** ``I0`` and ``rs`` are non-negative, and ``B`` is bounded
to ``[0, max signal]`` because a magnitude MR noise floor cannot be negative.
A voxel whose data would prefer ``B < 0`` returns ``B = 0``, which is Model A —
and ``delta_R2`` will show the baseline bought nothing there.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

#: Names of the parameter maps written for every dataset, in the requested order.
MAP_NAMES = (
    "I0_map_noB",
    "rs_map_noB",
    "I0_map_B",
    "rs_map_B",
    "B_map",
    "R2_noB_map",
    "R2_B_map",
    "RMSE_noB_map",
    "RMSE_B_map",
    "delta_R2_map",
    "model_choice_map",
    "fit_success_mask",
)

#: Per-voxel quantities carried through the fit, in CSV column order.
RESULT_FIELDS = (
    "I0_noB", "rs_noB",
    "I0_B", "rs_B", "B",
    "R2_noB", "R2_B",
    "RMSE_noB", "RMSE_B",
    "delta_R2",
    "AIC_noB", "AIC_B",
    "model_selected",
    "success_noB", "success_B",
    "fit_success",
    "n_points",
)

#: Encoding of ``model_choice_map`` / ``model_selected``.
MODEL_NONE, MODEL_NO_BASELINE, MODEL_WITH_BASELINE = 0, 1, 2

#: Upper bound on the relaxation rate, in 1/ms. T2 of 1 ms is already far
#: shorter than anything these acquisitions can resolve.
MAX_RATE = 1.0

#: Voxels need more points than the model has parameters for the fit to mean
#: anything; these are the minimum counts for the two models.
MIN_POINTS_NO_BASELINE = 3
MIN_POINTS_WITH_BASELINE = 4


def model_no_baseline(te, I0, rs):
    """Model A: ``I0 * exp(-rs * TE)``."""
    return I0 * np.exp(-rs * te)


def model_with_baseline(te, I0, rs, B):
    """Model B: ``I0 * exp(-rs * TE) + B``."""
    return I0 * np.exp(-rs * te) + B


# ---------------------------------------------------------------------------
# Single-voxel fitting
# ---------------------------------------------------------------------------


def _rate_guess(te: np.ndarray, signal: np.ndarray) -> float:
    """Initial decay rate from a log-linear regression.

    Fitting ``log(I)`` against ``TE`` is a different estimator from least
    squares on ``I``, so it is used only to start the real fit near the answer;
    a good start is what keeps the non-linear solver to a few iterations.
    """
    positive = signal > 0
    if positive.sum() < 2:
        return 0.01
    slope = np.polyfit(te[positive], np.log(signal[positive]), 1)[0]
    return float(np.clip(-slope, 1e-5, MAX_RATE * 0.5))


def _metrics(signal: np.ndarray, predicted: np.ndarray, n_params: int) -> tuple[float, float, float]:
    """Return ``(R2, RMSE, AIC)`` for one fit."""
    residual = signal - predicted
    rss = float(np.sum(residual**2))
    total = float(np.sum((signal - signal.mean()) ** 2))
    r2 = 1.0 - rss / total if total > 0 else 0.0
    rmse = float(np.sqrt(rss / signal.size))
    # Guard the log against a perfect fit.
    aic = signal.size * np.log(max(rss, 1e-30) / signal.size) + 2 * n_params
    return r2, rmse, float(aic)


def fit_one_voxel(te: np.ndarray, signal: np.ndarray) -> dict[str, float]:
    """Fit both models to one voxel's decay curve.

    Non-finite samples are dropped before fitting, and the voxel is skipped
    when too few remain, when the signal is flat, or when it never rises above
    zero — none of those carry a decay to measure.

    Args:
        te: Echo times, in milliseconds.
        signal: Intensity at each echo time, same length as ``te``.

    Returns:
        A dict with every key in :data:`RESULT_FIELDS`. A voxel that could not
        be fitted comes back with zeros and ``fit_success = 0`` rather than
        raising, so one bad voxel never stops a volume.
    """
    result = {name: 0.0 for name in RESULT_FIELDS}
    result["model_selected"] = MODEL_NONE

    finite = np.isfinite(te) & np.isfinite(signal)
    te_valid = np.asarray(te, dtype=np.float64)[finite]
    values = np.asarray(signal, dtype=np.float64)[finite]
    result["n_points"] = int(values.size)

    if values.size < MIN_POINTS_NO_BASELINE:
        return result
    peak = float(values.max())
    if peak <= 0 or np.ptp(values) == 0:
        return result  # no signal, or nothing to decay

    rate0 = _rate_guess(te_valid, values)
    floor = max(float(values.min()), 0.0)

    # --- Model A -----------------------------------------------------------
    try:
        params, _ = curve_fit(
            model_no_baseline, te_valid, values,
            p0=[peak, rate0],
            bounds=([0.0, 0.0], [10.0 * peak, MAX_RATE]),
            maxfev=5000,
        )
        if np.all(np.isfinite(params)):
            r2, rmse, aic = _metrics(
                values, model_no_baseline(te_valid, *params), n_params=2
            )
            result.update(
                I0_noB=float(params[0]), rs_noB=float(params[1]),
                R2_noB=r2, RMSE_noB=rmse, AIC_noB=aic, success_noB=1.0,
            )
    except (RuntimeError, ValueError, TypeError):
        pass  # non-convergent or degenerate; left at zero, flagged below

    # --- Model B -----------------------------------------------------------
    if values.size >= MIN_POINTS_WITH_BASELINE:
        try:
            params, _ = curve_fit(
                model_with_baseline, te_valid, values,
                # I0 starts as the amplitude *above* the floor, which is what
                # this parameter means once B is free.
                p0=[max(peak - floor, peak * 1e-3), rate0, floor],
                bounds=([0.0, 0.0, 0.0], [10.0 * peak, MAX_RATE, peak]),
                maxfev=5000,
            )
            if np.all(np.isfinite(params)):
                r2, rmse, aic = _metrics(
                    values, model_with_baseline(te_valid, *params), n_params=3
                )
                result.update(
                    I0_B=float(params[0]), rs_B=float(params[1]), B=float(params[2]),
                    R2_B=r2, RMSE_B=rmse, AIC_B=aic, success_B=1.0,
                )
        except (RuntimeError, ValueError, TypeError):
            pass

    if result["success_noB"] and result["success_B"]:
        result["delta_R2"] = result["R2_B"] - result["R2_noB"]
        result["fit_success"] = 1.0
        # Lower AIC wins; a tie goes to the model with fewer parameters.
        result["model_selected"] = (
            MODEL_WITH_BASELINE
            if result["AIC_B"] < result["AIC_noB"]
            else MODEL_NO_BASELINE
        )
    elif result["success_noB"]:
        result["model_selected"] = MODEL_NO_BASELINE
    elif result["success_B"]:
        result["model_selected"] = MODEL_WITH_BASELINE

    return result


# ---------------------------------------------------------------------------
# Many voxels
# ---------------------------------------------------------------------------


def _fit_chunk(payload):
    """Worker: fit a block of voxels. Module level so it can be pickled."""
    te, signals = payload
    output = np.zeros((signals.shape[0], len(RESULT_FIELDS)), dtype=np.float64)
    for row in range(signals.shape[0]):
        result = fit_one_voxel(te, signals[row])
        for column, name in enumerate(RESULT_FIELDS):
            output[row, column] = result[name]
    return output


def fit_signals(
    te: np.ndarray,
    signals: np.ndarray,
    workers: int | None = None,
    chunk_size: int = 4000,
    progress=None,
) -> dict[str, np.ndarray]:
    """Fit every row of ``signals`` independently.

    Args:
        te: Echo times in milliseconds, shape ``(n_echoes,)``.
        signals: Intensities, shape ``(n_voxels, n_echoes)``.
        workers: Processes to use. Defaults to all cores but one, so the
            machine stays usable. 1 runs in-process, which is easier to debug.
        chunk_size: Voxels per task.
        progress: Optional ``(done, total, message)`` callable.

    Returns:
        ``{field: array of shape (n_voxels,)}`` for every field in
        :data:`RESULT_FIELDS`.
    """
    signals = np.asarray(signals, dtype=np.float64)
    te = np.asarray(te, dtype=np.float64)
    n_voxels = signals.shape[0]
    if n_voxels == 0:
        return {name: np.zeros(0) for name in RESULT_FIELDS}

    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 2)

    chunks = [
        (te, signals[start:start + chunk_size])
        for start in range(0, n_voxels, chunk_size)
    ]
    collected: list[np.ndarray] = []

    if workers == 1:
        for done, chunk in enumerate(chunks, start=1):
            collected.append(_fit_chunk(chunk))
            if progress is not None:
                progress(done, len(chunks), "fitting")
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for done, block in enumerate(pool.map(_fit_chunk, chunks), start=1):
                collected.append(block)
                if progress is not None:
                    progress(done, len(chunks), "fitting")

    table = np.vstack(collected)
    return {name: table[:, column] for column, name in enumerate(RESULT_FIELDS)}


# ---------------------------------------------------------------------------
# Maps and output
# ---------------------------------------------------------------------------


@dataclass
class FitReport:
    """Summary of one dataset's fit."""

    name: str
    source: str
    output_dir: Path
    shape: tuple[int, int, int]
    echo_times: list[float] = field(default_factory=list)
    n_voxels: int = 0
    n_success: int = 0
    n_failed: int = 0
    model_counts: dict[str, int] = field(default_factory=dict)
    stats: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def failure_rate(self) -> float:
        """Fraction of evaluated voxels whose fit did not succeed."""
        return self.n_failed / self.n_voxels if self.n_voxels else 0.0

    @property
    def summary(self) -> str:
        """One-line human summary."""
        return (
            f"{self.name}: {self.n_voxels:,} voxels, "
            f"{self.n_success:,} fitted ({100 * (1 - self.failure_rate):.2f}%), "
            f"model A {self.model_counts.get('no_baseline', 0):,} / "
            f"model B {self.model_counts.get('with_baseline', 0):,}"
        )


def build_maps(
    results: dict[str, np.ndarray],
    coordinates: tuple[np.ndarray, np.ndarray, np.ndarray],
    shape: tuple[int, int, int],
) -> dict[str, np.ndarray]:
    """Scatter per-voxel results back into full-size 3D maps.

    Every map keeps the original ``(X, Y, Z)`` shape. Voxels that were not
    evaluated, and those whose fit failed, stay 0.
    """
    maps: dict[str, np.ndarray] = {}
    pairs = {
        "I0_map_noB": "I0_noB",
        "rs_map_noB": "rs_noB",
        "I0_map_B": "I0_B",
        "rs_map_B": "rs_B",
        "B_map": "B",
        "R2_noB_map": "R2_noB",
        "R2_B_map": "R2_B",
        "RMSE_noB_map": "RMSE_noB",
        "RMSE_B_map": "RMSE_B",
        "delta_R2_map": "delta_R2",
    }
    for map_name, field_name in pairs.items():
        volume = np.zeros(shape, dtype=np.float32)
        volume[coordinates] = results[field_name].astype(np.float32)
        maps[map_name] = volume

    choice = np.zeros(shape, dtype=np.int8)
    choice[coordinates] = results["model_selected"].astype(np.int8)
    maps["model_choice_map"] = choice

    success = np.zeros(shape, dtype=np.uint8)
    success[coordinates] = results["fit_success"].astype(np.uint8)
    maps["fit_success_mask"] = success
    return maps


def _statistics(values: np.ndarray) -> dict[str, float]:
    """Mean/median/std/percentiles of the successful fits of one parameter."""
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return {}
    return {
        "n": int(clean.size),
        "mean": float(clean.mean()),
        "median": float(np.median(clean)),
        "std": float(clean.std()),
        "min": float(clean.min()),
        "max": float(clean.max()),
        "p05": float(np.percentile(clean, 5)),
        "p95": float(np.percentile(clean, 95)),
    }


def write_fit_outputs(
    output_dir: str | Path,
    maps: dict[str, np.ndarray],
    coordinates: tuple[np.ndarray, np.ndarray, np.ndarray],
    results: dict[str, np.ndarray],
    metadata: dict,
) -> Path:
    """Write the maps, the per-voxel CSV and the metadata for one dataset.

    Maps are saved at full spatial size and never flattened; the CSV holds only
    the voxels that were actually evaluated.
    """
    import pandas as pd

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    for name in MAP_NAMES:
        np.save(directory / f"{name}.npy", maps[name])

    frame = pd.DataFrame(
        {
            # Columns are named x/y/z; arrays are indexed [z, y, x].
            "x": coordinates[2].astype(np.int32),
            "y": coordinates[1].astype(np.int32),
            "z": coordinates[0].astype(np.int32),
            "I0_noB": results["I0_noB"],
            "rs_noB": results["rs_noB"],
            "I0_B": results["I0_B"],
            "rs_B": results["rs_B"],
            "B": results["B"],
            "R2_noB": results["R2_noB"],
            "R2_B": results["R2_B"],
            "RMSE_noB": results["RMSE_noB"],
            "RMSE_B": results["RMSE_B"],
            "delta_R2": results["delta_R2"],
            "AIC_noB": results["AIC_noB"],
            "AIC_B": results["AIC_B"],
            "model_selected": results["model_selected"].astype(np.int8),
            "success_noB": results["success_noB"].astype(np.int8),
            "success_B": results["success_B"].astype(np.int8),
            "fit_success": results["fit_success"].astype(np.int8),
            "n_points": results["n_points"].astype(np.int16),
        }
    )
    # Six significant figures: the inputs are float32, so more digits would be
    # noise, and it roughly halves a file that runs to hundreds of megabytes.
    frame.to_csv(directory / "voxel_fit_data.csv", index=False, float_format="%.6g")

    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return directory


def summarise(
    name: str,
    source: str,
    output_dir: Path,
    shape: tuple[int, int, int],
    echo_times: list[float],
    results: dict[str, np.ndarray],
) -> FitReport:
    """Collect counts and parameter statistics over the successful fits."""
    success = results["fit_success"] > 0
    report = FitReport(
        name=name,
        source=source,
        output_dir=output_dir,
        shape=tuple(int(n) for n in shape),
        echo_times=[float(t) for t in echo_times],
        n_voxels=int(results["fit_success"].size),
        n_success=int(success.sum()),
        n_failed=int((~success).sum()),
        model_counts={
            "no_baseline": int((results["model_selected"] == MODEL_NO_BASELINE).sum()),
            "with_baseline": int((results["model_selected"] == MODEL_WITH_BASELINE).sum()),
            "none": int((results["model_selected"] == MODEL_NONE).sum()),
        },
    )
    for key in ("I0_noB", "rs_noB", "I0_B", "rs_B", "B", "R2_noB", "R2_B", "delta_R2"):
        report.stats[key] = _statistics(results[key][success])
    return report
