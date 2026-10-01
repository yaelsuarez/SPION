#!/usr/bin/env python3
"""Generate the Model A / Model B diagnostic figures and print the interpretation.

    python run_diagnostics.py

Reads the finished fits, writes PNGs and the summary workbook under
``i0_rs/model_diagnostics``. Nothing under ``i0_rs/Segmentations`` is modified.
"""

import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_ROOT  # noqa: E402

import numpy as np  # noqa: E402

MRI = DATA_ROOT
FIT_ROOT = MRI / "i0_rs" / "Segmentations"
SEG_ROOT = MRI / "Segmentations"
OUT_ROOT = MRI / "i0_rs" / "model_diagnostics"


def peak_ram_gb() -> float:
    """Peak resident set size of this process, in GB (macOS reports bytes)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def main() -> int:
    import echoviewer  # noqa: F401
    from echoviewer import diagnostics as dg

    started = time.time()
    pools, sample_pools, per_segment, matrices, labels, tissue_stats, saved = dg.run(
        FIT_ROOT, SEG_ROOT, OUT_ROOT
    )

    pngs = [p for p in saved if p.suffix == ".png"]
    every = {name: np.concatenate([pools[l][name] for l in labels]) for name in
             ("rs_noB", "rs_B", "I0_noB", "I0_B", "B", "delta_AIC", "model_selected")}
    n = every["model_selected"].size

    print("\n" + "=" * 78)
    print("INTERPRETATION  (all numbers from the complete valid voxel population)")
    print("=" * 78)

    print("\nA. MODEL COMPARISON")
    frac_a = float((every["model_selected"] == 1).mean())
    frac_b = float((every["model_selected"] == 2).mean())
    print(f"   Model A preferred: {100*frac_a:5.1f}%   Model B preferred: {100*frac_b:5.1f}%"
          f"   (AIC, {n:,} voxels)")
    print("   By material (% preferring Model B):")
    for label in labels:
        pool = pools[label]
        print(f"     {label:7s} {100*float((pool['model_selected'] == 2).mean()):5.1f}%"
              f"   median B = {np.median(pool['B']):7.2f}   n = {pool['rs_B'].size:,}")
    print(f"   B over all voxels: median {np.median(every['B']):.2f}, "
          f"{100*float((every['B'] > 1).mean()):.1f}% above 1 -> "
          f"{'substantially non-zero' if np.median(every['B']) > 1 else 'close to zero'}")

    print("\nB. PARAMETER STABILITY")
    d_rs = every["rs_B"] - every["rs_noB"]
    d_i0 = every["I0_B"] - every["I0_noB"]
    print(f"   rs: median change {np.median(d_rs):+.5f} 1/ms "
          f"({100*np.median(d_rs/np.maximum(every['rs_noB'], 1e-9)):+.1f}% of rs_noB)")
    print(f"   I0: median change {np.median(d_i0):+.1f} a.u.")
    print("   Per material (median delta_rs):")
    for label in labels:
        pool = pools[label]
        print(f"     {label:7s} {np.median(pool['rs_B'] - pool['rs_noB']):+.5f}")

    print("\nC. CONCENTRATION DISCRIMINATION (rs_B)")
    for label in labels:
        pool = pools[label]
        q1, q3 = dg.iqr(pool["rs_B"])
        print(f"     {label:7s} median {np.median(pool['rs_B']):.5f}  IQR [{q1:.5f}, {q3:.5f}]")
    matrix = matrices["rs_B"]
    worst = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            worst.append((matrix[i, j], labels[i], labels[j]))
    worst.sort(reverse=True)
    print("   Worst overlaps using rs_B alone:")
    for value, first, second in worst[:4]:
        print(f"     {first:7s} vs {second:7s}  overlap {value:.3f}")

    print("\nD. TISSUE vs 0.1 mg/mL")
    for key in ("rs_B", "rs_B_histogram", "I0_B", "B", "rs_B+I0_B", "rs_B+I0_B+B"):
        if key in tissue_stats:
            print(f"     {key:15s} overlap {tissue_stats[key]:.3f}")

    print("\nE. CALIBRATION CONSISTENCY (median rs_B per sample and concentration)")
    conc_labels = [l for l in labels if l in dg.CONCENTRATIONS]
    header = "     " + "conc   " + "".join(f"{s:>12s}" for s in sorted(sample_pools))
    print(header)
    for label in conc_labels:
        line = f"     {label:6s} "
        for sample in sorted(sample_pools):
            pool = sample_pools[sample].get(label)
            line += f"{np.median(pool['rs_B']):12.5f}" if pool is not None else f"{'-':>12s}"
        print(line)

    print("\nF. RECOMMENDATION")
    one = tissue_stats.get("rs_B", float("nan"))
    two = tissue_stats.get("rs_B+I0_B", float("nan"))
    three = tissue_stats.get("rs_B+I0_B+B", float("nan"))
    if three < two * 0.9:
        pick = "rs_B + I0_B + B"
    elif two < one * 0.9:
        pick = "rs_B + I0_B"
    else:
        pick = "rs_B"
    print(f"   tissue/0.1 overlap: rs_B {one:.3f} -> +I0_B {two:.3f} -> +B {three:.3f}")
    print(f"   => start the classifier with: {pick}")

    print("\n" + "=" * 78)
    print(f"Diagnostics directory : {OUT_ROOT}")
    print(f"PNGs generated        : {len(pngs)}")
    print(f"Workbook              : {OUT_ROOT / 'summary' / 'diagnostic_summary.xlsx'}")
    print(f"Peak RAM              : {peak_ram_gb():.2f} GB")
    print(f"Elapsed               : {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
