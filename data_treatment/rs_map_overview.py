#!/usr/bin/env python3
"""rs-only overview for the training samples, peak-rs slice in each orientation.

One row per sample, three orthogonal views, nothing overlaid. Slice choice is
literal: for each orientation the slice holding the highest rs value. That is
safe here because no sample approaches the 1.0 fit bound (maxima are 0.31-0.41),
so the peak is real signal rather than a failed fit; the three planes therefore
all pass through the single peak voxel.

rs is the AIC-selected rate, rebuilt from the stored maps exactly as everywhere
else in the project: rs_map_noB where model_choice is 1, rs_map_B otherwise.
Nothing is refitted and no original file is written to.
"""

import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MRI = DATA_ROOT
SRC = MRI / "i0_rs" / "Volumes"
OUT = MRI / "Results" / "rs_map"
TRAINING = ["Syringes", "3D", "Pill1"]
VIEWS = ["Axial", "Coronal", "Sagittal"]
NAME = "rs_map_overview"


def plane(volume, view, index):
    """dicom_viewer's convention: axial as stored, coronal and sagittal flipped."""
    if view == "Axial":
        return volume[index]
    if view == "Coronal":
        return np.flipud(volume[:, index, :])
    return np.flipud(volume[:, :, index])


def main(samples=None, rule="peak", out_dir=None, name=None, copy_to_source=True) -> int:
    """rule: 'peak' picks the slice holding the highest rs, 'middle' the centre slice."""
    from echoviewer.diagnostics import _figure

    started = time.time()
    samples = samples or TRAINING
    out_dir = out_dir or OUT
    name = name or NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    volumes, report = {}, []
    # every training sample is loaded even when only some are drawn, so the
    # colour scale stays identical between figures
    for sample in TRAINING:
        valid = np.load(SRC / sample / "fit_success_mask.npy") > 0
        choice = np.load(SRC / sample / "model_choice_map.npy")
        rs = np.where(choice == 1, np.load(SRC / sample / "rs_map_noB.npy"),
                      np.load(SRC / sample / "rs_map_B.npy")).astype(np.float32)
        rs = np.where(valid & np.isfinite(rs), rs, np.nan)
        volumes[sample] = rs
        print(f"  {sample:9s} {rs.shape}  valid {int(valid.sum()):,}  "
              f"max rs {np.nanmax(rs):.4f} 1/ms")

    everything = np.concatenate([v[np.isfinite(v)] for v in volumes.values()])
    vmax = float(np.percentile(everything, 99.9))
    print(f"\nshared colour scale 0 to {vmax:.3f} 1/ms (99.9th percentile of all "
          f"training voxels); higher values saturate")

    picks = {}
    for sample in samples:
        rs = volumes[sample]
        if rule == "middle":
            picks[sample] = dict(zip(VIEWS, (d // 2 for d in rs.shape)))
        else:
            filled = np.nan_to_num(rs, nan=-np.inf)
            picks[sample] = {"Axial": int(filled.max(axis=(1, 2)).argmax()),
                             "Coronal": int(filled.max(axis=(0, 2)).argmax()),
                             "Sagittal": int(filled.max(axis=(0, 1)).argmax())}
        for view in VIEWS:
            index = picks[sample][view]
            sl = plane(rs, view, index)
            report.append({"sample": sample, "view": view, "slice_index": index,
                           "slice_max_rs": float(np.nanmax(sl)),
                           "slice_mean_rs": float(np.nanmean(sl)),
                           "valid_voxels_in_slice": int(np.isfinite(sl).sum())})
        print(f"  {sample:9s} {rule} slices " +
              "  ".join(f"{v} {picks[sample][v]}" for v in VIEWS))

    fig = _figure(figsize=(13.5, 4.3 * len(samples) + 1.2))
    axes = fig.subplots(len(samples), len(VIEWS), squeeze=False)
    im = None
    for r, sample in enumerate(samples):
        for c, view in enumerate(VIEWS):
            index = picks[sample][view]
            ax = axes[r][c]
            im = ax.imshow(plane(volumes[sample], view, index), cmap="inferno",
                           vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_facecolor("black")
            ax.set_xticks([]); ax.set_yticks([])
            label = "middle" if rule == "middle" else "peak-rs"
            ax.set_title(f"{view} — {label} slice {index}", fontsize=10)
            if c == 0:
                ax.set_ylabel(sample, fontsize=12)
    bar = fig.colorbar(im, cax=fig.add_axes([0.92, 0.08, 0.013, 0.84]))
    bar.set_label(f"rs (1/ms), shared scale 0–{vmax:.3f}; higher values saturate",
                  fontsize=10)
    what = ("middle slice of each orientation" if rule == "middle"
            else "slice with the highest rs in each orientation")
    fig.suptitle(f"rs map{'s' if len(samples) > 1 else ''}, "
                 f"{', '.join(samples)} — {what}", fontsize=14)

    written = []
    for suffix in (".svg", ".png"):          # SVG first; the PNG write clears the figure
        path = out_dir / f"{name}{suffix}"
        fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
        written.append(path)
    fig.clf()

    frame = pd.DataFrame(report)
    frame.round(6).to_csv(out_dir / f"{name}_slice_selection.csv", index=False)
    (out_dir / f"{name}_README.json").write_text(json.dumps({
        "source": str(SRC.relative_to(MRI)),
        "samples": samples,
        "quantity": ("AIC-selected rs: rs_map_noB where model_choice == 1, rs_map_B "
                     "otherwise; masked to fit_success_mask"),
        "slice_rule": ("centre index of each orientation" if rule == "middle"
                       else "per orientation, the slice containing the highest rs value"),
        "colour_scale": {"vmin": 0.0, "vmax": vmax,
                         "basis": ("99.9th percentile over all training voxels, shared with "
                                   "the other rs overviews")},
        "plane_convention": "dicom_viewer: axial as stored, coronal and sagittal flipped",
        "overlays": "none; rs only",
        "originals_modified": False,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2, default=float))

    if copy_to_source:      # alongside the source volumes, as requested
        for path in written:
            shutil.copy2(path, SRC / path.name)
    print("\n" + frame.to_string(index=False))
    print(f"\nsaved to {out_dir}" + (f" and copied to {SRC}" if copy_to_source else ""))
    print(f"elapsed {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
