#!/usr/bin/env python3
"""Volume error, Brier, MCC, HD95 and ASSD on Pill2 from the saved predictions.

No classifier is retrained, refitted, tuned or modified. Every number derives
from the existing per-classifier pill2_segmented_predictions.csv.gz over the
same 384,315 labelled voxels used throughout.

Distance metrics run on the real 3D voxel grid at the acquisition spacing, read
from the Pill2 DICOM headers: PixelSpacing 0.32 x 0.32 mm, SliceThickness
0.32 mm, and a measured slice-position step of exactly 0.3200 mm - isotropic.

Degenerate cases are reported as blank with a status string rather than as a
number. sklearn returns MCC = 0.0 when a class is absent from both the truth
and the prediction, which is indistinguishable from genuinely chance-level
agreement; that case is detected and blanked instead.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = DATA_ROOT / "Classifiers"
SRC = ROOT / "Classifiers"
DST = ROOT / "Metrics"
ORDER = ["logistic_regression", "random_forest", "gradient_boosting", "rbf_svm", "knn",
         "lda", "qda", "gmm", "hierarchical", "ordinal"]
CLASSES = ["0", "0.05", "0.1", "0.2", "0.25", "0.3", "tissue"]
SHAPE = (128, 330, 128)                 # z (slice), y (row), x (column)
SPACING = (0.32, 0.32, 0.32)            # mm, isotropic
VOXEL_MM3 = float(np.prod(SPACING))


def surface(mask, structure):
    from scipy.ndimage import binary_erosion
    return mask & ~binary_erosion(mask, structure=structure, border_value=0)


def distance_pair(true_mask, pred_mask):
    """Symmetric surface distances in mm, or None if either mask is empty."""
    from scipy.ndimage import distance_transform_edt, generate_binary_structure
    if not true_mask.any() or not pred_mask.any():
        return None
    st = generate_binary_structure(3, 1)
    ts, ps = surface(true_mask, st), surface(pred_mask, st)
    if not ts.any() or not ps.any():
        return None
    d_to_true = distance_transform_edt(~ts, sampling=SPACING)
    d_to_pred = distance_transform_edt(~ps, sampling=SPACING)
    return np.concatenate([d_to_true[ps], d_to_pred[ts]])


def main() -> int:
    from sklearn.metrics import matthews_corrcoef
    from echoviewer.diagnostics import _figure, _save

    started = time.time()
    for folder in ("Volume_error", "Brier", "MCC", "HD95", "ASSD"):
        assert not (DST / folder).exists(), f"{folder} already exists; refusing to overwrite"
    assert not (DST / "summary.csv").is_file(), "Metrics/summary.csv already exists"

    volume_rows, brier_rows, mcc_rows, dist_rows, combined = [], [], [], [], []

    for name in ORDER:
        d = pd.read_csv(SRC / name / "pill2_segmented_predictions.csv.gz",
                        dtype={"material": str, "predicted": str})
        truth = d.material.to_numpy()
        pred = d.predicted.to_numpy()
        proba = d[[f"p_{c}" for c in CLASSES]].to_numpy(np.float64)
        z = d.z.to_numpy(np.intp); y = d.y.to_numpy(np.intp); x = d.x.to_numpy(np.intp)
        n = len(d)

        # ---- multiclass MCC and multiclass Brier --------------------------
        onehot = np.zeros_like(proba)
        for i, c in enumerate(CLASSES):
            onehot[:, i] = (truth == c)
        multiclass_brier = float(((proba - onehot) ** 2).sum(1).mean())
        mcc_all = float(matthews_corrcoef(truth, pred))

        per_class_brier, per_class_mcc, hd95s, assds = {}, {}, {}, {}
        for i, c in enumerate(CLASSES):
            t_mask_flat = truth == c
            p_mask_flat = pred == c
            n_true, n_pred = int(t_mask_flat.sum()), int(p_mask_flat.sum())

            # ---- volume error ---------------------------------------------
            diff = n_pred - n_true
            rel = (diff / n_true) if n_true else np.nan
            volume_rows.append({
                "classifier": name, "class": c,
                "n_true_voxels": n_true, "n_predicted_voxels": n_pred,
                "true_volume_mm3": n_true * VOXEL_MM3,
                "predicted_volume_mm3": n_pred * VOXEL_MM3,
                "signed_error_voxels": diff,
                "absolute_error_voxels": abs(diff),
                "absolute_error_mm3": abs(diff) * VOXEL_MM3,
                "relative_error": rel,
                "relative_error_pct": rel * 100 if n_true else np.nan,
                "status": "ok" if n_true else "no ground-truth voxels: relative error undefined",
            })

            # ---- Brier, defined even with no positives ---------------------
            b = float(((proba[:, i] - t_mask_flat) ** 2).mean())
            per_class_brier[c] = b
            brier_rows.append({"classifier": name, "class": c, "brier": b,
                               "n_true_voxels": n_true,
                               "mean_probability": float(proba[:, i].mean()),
                               "prevalence": n_true / n,
                               "status": "ok" if n_true else
                                         "no ground-truth voxels: measures false confidence only"})

            # ---- one-vs-rest MCC -------------------------------------------
            if n_true == 0 and n_pred == 0:
                m, status = np.nan, "class absent from both truth and prediction: MCC undefined"
            elif n_true == 0 or n_pred == 0 or n_true == n or n_pred == n:
                m, status = 0.0, "one side is constant: MCC is exactly 0 by definition"
            else:
                m, status = float(matthews_corrcoef(t_mask_flat, p_mask_flat)), "ok"
            per_class_mcc[c] = m
            mcc_rows.append({"classifier": name, "class": c, "mcc_ovr": m,
                             "n_true_voxels": n_true, "n_predicted_voxels": n_pred,
                             "status": status})

            # ---- HD95 and ASSD on the 3D masks ------------------------------
            if n_true == 0 or n_pred == 0:
                hd, assd = np.nan, np.nan
                status = ("no ground-truth voxels" if n_true == 0
                          else "class never predicted")
            else:
                tv = np.zeros(SHAPE, bool); tv[z[t_mask_flat], y[t_mask_flat], x[t_mask_flat]] = True
                pv = np.zeros(SHAPE, bool); pv[z[p_mask_flat], y[p_mask_flat], x[p_mask_flat]] = True
                dist = distance_pair(tv, pv)
                if dist is None:
                    hd, assd, status = np.nan, np.nan, "no extractable surface"
                else:
                    hd = float(np.percentile(dist, 95))
                    assd = float(dist.mean())
                    status = "ok"
                del tv, pv
            hd95s[c], assds[c] = hd, assd
            dist_rows.append({"classifier": name, "class": c, "hd95_mm": hd, "assd_mm": assd,
                              "n_true_voxels": n_true, "n_predicted_voxels": n_pred,
                              "spacing_mm": "0.32 x 0.32 x 0.32", "status": status})

        defined = lambda dd: [v for v in dd.values() if np.isfinite(v)]
        vol = pd.DataFrame([r for r in volume_rows if r["classifier"] == name])
        combined.append({
            "classifier": name,
            "mcc_multiclass": mcc_all,
            "mcc_ovr_macro": float(np.mean(defined(per_class_mcc))),
            "brier_multiclass": multiclass_brier,
            "brier_macro": float(np.mean(list(per_class_brier.values()))),
            "hd95_macro_mm": float(np.mean(defined(hd95s))) if defined(hd95s) else np.nan,
            "assd_macro_mm": float(np.mean(defined(assds))) if defined(assds) else np.nan,
            "mean_abs_relative_volume_error": float(np.nanmean(np.abs(vol.relative_error))),
            "total_abs_volume_error_mm3": float(vol.absolute_error_mm3.sum()),
            "classes_with_distance_metrics": len(defined(hd95s)),
            "n_voxels": n,
        })
        print(f"  {name:20s} MCC {mcc_all:.4f}  Brier {multiclass_brier:.4f}  "
              f"HD95 {combined[-1]['hd95_macro_mm']:.2f} mm  "
              f"ASSD {combined[-1]['assd_macro_mm']:.2f} mm  "
              f"({time.time()-started:.0f}s)")
        del d, proba, onehot

    # ---- write ------------------------------------------------------------
    def write(folder, frame, filename):
        out = DST / folder
        out.mkdir(parents=True, exist_ok=True)
        frame.round(6).to_csv(out / filename, index=False)
        return out

    vol = pd.DataFrame(volume_rows)
    brier = pd.DataFrame(brier_rows)
    mcc = pd.DataFrame(mcc_rows)
    dist = pd.DataFrame(dist_rows)
    write("Volume_error", vol, "volume_error.csv")
    write("Brier", brier, "brier.csv")
    write("MCC", mcc, "mcc.csv")
    write("HD95", dist[["classifier", "class", "hd95_mm", "n_true_voxels",
                        "n_predicted_voxels", "spacing_mm", "status"]], "hd95.csv")
    write("ASSD", dist[["classifier", "class", "assd_mm", "n_true_voxels",
                        "n_predicted_voxels", "spacing_mm", "status"]], "assd.csv")
    summary = pd.DataFrame(combined)
    summary.round(6).to_csv(DST / "summary.csv", index=False)

    def heat(frame, value, title, path, cmap="viridis", fmt="{:.2f}", vmin=None, vmax=None):
        table = frame.pivot(index="classifier", columns="class", values=value).reindex(ORDER)
        table = table[[c for c in CLASSES if c in table.columns]]
        fig = _figure(figsize=(1.45 * table.shape[1] + 5, 0.55 * table.shape[0] + 2.6))
        ax = fig.subplots()
        values = table.to_numpy(np.float64)
        finite = values[np.isfinite(values)]
        im = ax.imshow(values, cmap=cmap,
                       vmin=vmin if vmin is not None else (finite.min() if finite.size else 0),
                       vmax=vmax if vmax is not None else (finite.max() if finite.size else 1),
                       aspect="auto")
        ax.set_xticks(range(table.shape[1])); ax.set_xticklabels(table.columns, fontsize=9)
        ax.set_yticks(range(table.shape[0])); ax.set_yticklabels(table.index, fontsize=9)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                v = values[i, j]
                ax.text(j, i, fmt.format(v) if np.isfinite(v) else "–", ha="center",
                        va="center", fontsize=7.5, color="white" if np.isfinite(v) and
                        v < (finite.min() + finite.max()) / 2 else "black")
        fig.colorbar(im, ax=ax, fraction=0.02)
        ax.set_title(title, fontsize=11)
        _save(fig, path, [])

    heat(vol, "relative_error_pct",
         "Volume error (%), predicted vs true Pill2 volume — “–” = no ground truth",
         DST / "Volume_error" / "volume_error.png", cmap="coolwarm", fmt="{:+.0f}",
         vmin=-200, vmax=200)
    heat(brier, "brier", "Brier score per class (lower is better)",
         DST / "Brier" / "brier.png", cmap="viridis_r", fmt="{:.3f}")
    heat(mcc, "mcc_ovr", "One-vs-rest MCC per class — “–” = class absent from truth and prediction",
         DST / "MCC" / "mcc.png", cmap="viridis", fmt="{:.3f}", vmin=0, vmax=1)
    heat(dist, "hd95_mm", "HD95 (mm), spacing 0.32 mm isotropic — “–” = undefined",
         DST / "HD95" / "hd95.png", cmap="magma_r", fmt="{:.1f}")
    heat(dist, "assd_mm", "ASSD (mm), spacing 0.32 mm isotropic — “–” = undefined",
         DST / "ASSD" / "assd.png", cmap="magma_r", fmt="{:.2f}")

    fig = _figure(figsize=(12, 5.8))
    ax = fig.subplots()
    s = summary.sort_values("mcc_multiclass", ascending=False)
    ax.bar(range(len(s)), s.mcc_multiclass, color="#4a6fe3")
    for i, v in enumerate(s.mcc_multiclass):
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(s))); ax.set_xticklabels(s.classifier, rotation=35, ha="right",
                                                     fontsize=9)
    ax.set_ylabel("multiclass MCC"); ax.grid(alpha=.25, axis="y")
    ax.set_title("Pill2 multiclass Matthews correlation coefficient")
    _save(fig, DST / "MCC" / "mcc_multiclass.png", [])

    (DST / "metric_notes.json").write_text(json.dumps({
        "source": "existing pill2_segmented_predictions.csv.gz; no model retrained or modified",
        "n_voxels": int(summary.n_voxels.iloc[0]),
        "grid_shape_zyx": list(SHAPE),
        "voxel_spacing_mm": list(SPACING),
        "spacing_source": ("Pill2 DICOM: PixelSpacing 0.32x0.32 mm, SliceThickness 0.32 mm, "
                           "measured ImagePositionPatient step 0.3200 mm - isotropic"),
        "voxel_volume_mm3": VOXEL_MM3,
        "hd95_assd_definition": ("symmetric: surface voxels of each mask (6-connected erosion), "
                                 "Euclidean distance transform at the spacing above, both "
                                 "directions pooled; HD95 = 95th percentile, ASSD = mean"),
        "classes_without_pill2_ground_truth": ["0.05", "0.2"],
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2, default=float))
    pd.set_option("display.width", 240)
    print("\n" + summary.round(4).to_string(index=False))
    print(f"\nelapsed {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
