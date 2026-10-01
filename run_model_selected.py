#!/usr/bin/env python3
"""Build the AIC-selected parameter representation and its I0-vs-rs figures.

    python run_model_selected.py

Reads the existing fits, writes PNGs into
``i0_rs/model_diagnostics/parameter_relationships``. Nothing is refitted and no
existing output is modified.
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
OUT_ROOT = MRI / "i0_rs" / "model_diagnostics"


def main() -> int:
    import echoviewer  # noqa: F401
    from echoviewer import model_selected as ms
    from echoviewer.diagnostics import SEED, iqr

    started = time.time()
    print("Collecting fits (no refitting; stored model_selected used as the AIC decision)")
    data = ms.collect(FIT_ROOT)

    saved: list[Path] = []
    print("\nFigures:")
    records = ms.figures(data, OUT_ROOT, saved)

    print("\n" + "=" * 76)
    print("REPORT")
    print("=" * 76)

    print("\n1. PNG files written")
    for path in saved:
        print(f"   {path}")

    print("\n2. Voxels per sample")
    for sample in data.samples:
        print(f"   {sample:9s} {data.count(sample):>10,}")
    print(f"   {'TOTAL':9s} {data.count():>10,}")

    chose_a = np.concatenate([v["chose_a"] for v in data.groups.values()])
    n_a, n_b = int((chose_a == 1).sum()), int((chose_a == 0).sum())
    print("\n3. Model selected (AIC; ties go to Model A)")
    print(f"   Model A: {n_a:>10,}  ({100 * n_a / chose_a.size:5.2f}%)")
    print(f"   Model B: {n_b:>10,}  ({100 * n_b / chose_a.size:5.2f}%)")

    print("\n4. Model A / B percentage by material")
    print(f"   {'material':9s} {'n':>10s} {'% A':>7s} {'% B':>7s}")
    for label in data.labels:
        pool = data.by_label(label)
        a = 100 * float((pool["chose_a"] == 1).mean())
        print(f"   {label:9s} {pool['chose_a'].size:>10,} {a:7.2f} {100 - a:7.2f}")
    print("\n   by sample")
    for sample in data.samples:
        flags = np.concatenate([v["chose_a"] for (s, _), v in data.groups.items() if s == sample])
        a = 100 * float((flags == 1).mean())
        print(f"   {sample:9s} {flags.size:>10,} {a:7.2f} {100 - a:7.2f}")

    print("\n5. Selected parameter ranges (complete voxel population)")
    print(f"   {'material':9s} {'rs_sel median':>14s} {'rs IQR':>22s} "
          f"{'I0_sel median':>14s} {'I0 IQR':>24s} {'B_sel median':>13s}")
    for label in data.labels:
        pool = data.by_label(label)
        rq1, rq3 = iqr(pool["rs_sel"])
        iq1, iq3 = iqr(pool["I0_sel"])
        print(f"   {label:9s} {np.median(pool['rs_sel']):14.5f} "
              f"[{rq1:9.5f},{rq3:9.5f}] {np.median(pool['I0_sel']):14.1f} "
              f"[{iq1:10.1f},{iq3:10.1f}] {np.median(pool['B_sel']):13.2f}")

    print("\n6-7. Concentration vs phantom dependence")
    print("   median I0_selected by sample and material (constant down a column")
    print("   means I0 tracks the phantom, not the material):")
    header = f"   {'material':9s}" + "".join(f"{s:>11s}" for s in data.samples)
    print(header)
    for label in data.labels:
        line = f"   {label:9s}"
        for sample in data.samples:
            pool = data.by_label(label, sample=sample)
            line += f"{np.median(pool['I0_sel']):11.0f}" if pool else f"{'-':>11s}"
        print(line)
    print("\n   median rs_selected by sample and material:")
    print(header)
    for label in data.labels:
        line = f"   {label:9s}"
        for sample in data.samples:
            pool = data.by_label(label, sample=sample)
            line += f"{np.median(pool['rs_sel']):11.5f}" if pool else f"{'-':>11s}"
        print(line)

    print("\n8. Does tissue overlap any SPION concentration in rs_selected?")
    tissue = data.by_label("tissue")
    tq1, tq3 = iqr(tissue["rs_sel"])
    print(f"   tissue rs_sel IQR [{tq1:.5f}, {tq3:.5f}]")
    for label in data.labels:
        if label == "tissue":
            continue
        pool = data.by_label(label)
        q1, q3 = iqr(pool["rs_sel"])
        overlaps = not (q3 < tq1 or q1 > tq3)
        print(f"     {label:7s} IQR [{q1:.5f}, {q3:.5f}]  "
              f"{'IQRs OVERLAP' if overlaps else 'separated'}")

    print("\n9. Is the selected representation different from A and B?")
    for key_a, key_sel, name in (("rs_A", "rs_sel", "rs"), ("I0_A", "I0_sel", "I0")):
        changed = []
        for values in data.groups.values():
            changed.append(values[key_a] != values[key_sel])
        changed = np.concatenate(changed)
        print(f"   {name}_selected differs from Model A in {100 * changed.mean():5.2f}% of voxels")
    for key_b, key_sel, name in (("rs_B", "rs_sel", "rs"), ("I0_B", "I0_sel", "I0")):
        changed = np.concatenate([v[key_b] != v[key_sel] for v in data.groups.values()])
        print(f"   {name}_selected differs from Model B in {100 * changed.mean():5.2f}% of voxels")

    print("\n10. Provenance of every scatter figure")
    for record in records:
        if record.plotted:
            print(f"   {record.name:45s} {record.plotted:>9,} / {record.available:>9,} "
                  f"({100 * record.fraction:5.1f}%) seed {record.seed}")
        else:
            print(f"   {record.name:45s} {'all voxels, no subsampling':>34s}")

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    print(f"\nPNGs: {len([p for p in saved if p.suffix == '.png'])}   "
          f"Peak RAM: {peak:.2f} GB   Elapsed: {time.time() - started:.0f} s")
    print(f"Directory: {OUT_ROOT / 'parameter_relationships'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
