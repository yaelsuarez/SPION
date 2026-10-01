#!/usr/bin/env python3
"""First and last echo of one sample, three orthogonal views, rs-overview style.

Slice indices are chosen once, on the first echo, by the same rule the rs
overview uses (the slice holding the highest value) and then reused for the last
echo, so the two rows show the same anatomy and the decay between them is the
only difference. The colour scale is shared for the same reason.

Raw signal is shown unmasked and unscaled: nothing is fitted, recalculated or
written back to the source volumes.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT, FIGURES  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data_treatment.rs_map_overview import VIEWS, plane  # noqa: E402

MRI = DATA_ROOT
FIGURES = FIGURES
SAMPLE = "Pill1"
NAME = f"{SAMPLE}_first_last_echo_overview"


def main() -> int:
    from echoviewer.dataset import get_available_echotimes, load_volume_from_folder
    from echoviewer.diagnostics import _figure

    started = time.time()
    FIGURES.mkdir(parents=True, exist_ok=True)

    echoes = get_available_echotimes(MRI / "Volumes" / SAMPLE)
    chosen = [echoes[0], echoes[-1]]
    volumes = {}
    for echo in chosen:
        volumes[echo.value] = load_volume_from_folder(echo.path).data.astype(np.float32)
        v = volumes[echo.value]
        print(f"  TE {echo.value:6g} ms  shape {v.shape}  max {v.max():8.1f}  "
              f"mean {v.mean():7.1f}")

    first, last = chosen[0].value, chosen[-1].value
    reference = volumes[first]
    picks = {"Axial": int(reference.max(axis=(1, 2)).argmax()),
             "Coronal": int(reference.max(axis=(0, 2)).argmax()),
             "Sagittal": int(reference.max(axis=(0, 1)).argmax())}
    print(f"\n  peak-signal slices from the {first:g} ms echo, reused for {last:g} ms: "
          + "  ".join(f"{v} {picks[v]}" for v in VIEWS))

    vmax = float(np.percentile(reference, 99.9))
    print(f"  shared colour scale 0 to {vmax:.0f} (99.9th percentile of the "
          f"{first:g} ms echo); higher values saturate")

    report = []
    fig = _figure(figsize=(13.5, 4.3 * len(chosen) + 1.2))
    axes = fig.subplots(len(chosen), len(VIEWS), squeeze=False)
    im = None
    for r, echo in enumerate(chosen):
        for c, view in enumerate(VIEWS):
            index = picks[view]
            sl = plane(volumes[echo.value], view, index)
            ax = axes[r][c]
            im = ax.imshow(sl, cmap="inferno", vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_facecolor("black")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{view} — slice {index}", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"TE = {echo.value:g} ms", fontsize=12)
            report.append({"sample": SAMPLE, "echo_time_ms": echo.value, "view": view,
                           "slice_index": index, "slice_max": float(sl.max()),
                           "slice_mean": float(sl.mean())})
    bar = fig.colorbar(im, cax=fig.add_axes([0.92, 0.08, 0.013, 0.84]))
    bar.set_label(f"signal intensity (a.u.), shared scale 0–{vmax:.0f}; "
                  "higher values saturate", fontsize=10)
    fig.suptitle(f"{SAMPLE} — first and last echo, peak-signal slice in each orientation "
                 f"(slices fixed by the {first:g} ms echo)", fontsize=14)

    written = []
    for suffix in (".svg", ".png"):        # SVG first; the PNG write clears the figure
        path = FIGURES / f"{NAME}{suffix}"
        fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
        written.append(path)
    fig.clf()

    frame = pd.DataFrame(report)
    frame.round(4).to_csv(FIGURES / f"{NAME}_slice_selection.csv", index=False)
    (FIGURES / f"{NAME}_README.json").write_text(json.dumps({
        "source": f"MRI/Volumes/{SAMPLE}/ — original echo-time volumes, unmodified",
        "echo_times_ms": [first, last],
        "quantity": "raw signal intensity, unmasked and unscaled",
        "slice_rule": (f"per orientation, the slice holding the highest signal in the "
                       f"{first:g} ms echo; the same indices are reused for {last:g} ms"),
        "colour_scale": {"vmin": 0.0, "vmax": vmax,
                         "basis": f"99.9th percentile of the {first:g} ms echo, shared "
                                  "between both rows"},
        "plane_convention": "dicom_viewer: axial as stored, coronal and sagittal flipped",
        "note": ("slice indices differ from rs_map_overview, which picked peak-rs slices; "
                 "these are peak-signal slices"),
        "recalculated": False, "originals_modified": False,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2, default=float))

    print("\n" + frame.to_string(index=False))
    print(f"\nsaved to {FIGURES}\nelapsed {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
