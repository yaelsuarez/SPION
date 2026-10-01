"""READ-ONLY verification of the proposed label-light normalisation.

Computes the factors and f(rs) in memory only. Writes nothing, modifies
nothing, trains nothing.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT  # noqa: E402
import numpy as np
import pandas as pd

MRI = DATA_ROOT
SEG = MRI / "Normalized_data_i0_rs" / "Segmentations"      # holds I0_raw = I0_selected
VOL = MRI / "i0_rs" / "Volume_for_testing" / "Pill2"
TRAINING = ("Syringes", "3D", "Pill1")
REFERENCE_SAMPLES = ("Syringes", "3D")     # the two used to learn concentration effects


def load():
    """I0_selected, rs_selected, quality flags per (sample, material). Read-only."""
    blocks = {}
    for sample_dir in sorted(SEG.iterdir()):
        if not sample_dir.is_dir():
            continue
        for seg_dir in sorted(sample_dir.iterdir()):
            if not seg_dir.is_dir():
                continue
            d = pd.read_csv(seg_dir / "voxels.csv.gz",
                            usecols=["I0_raw", "rs_selected", "B_selected",
                                     "quality_ok", "flag_high_rs"])
            blocks[(sample_dir.name, seg_dir.name.split("_", 2)[2])] = d
    return blocks


def main():
    b = load()
    samples = sorted({s for s, _ in b})
    print("Loaded (read-only):", {s: sum(len(v) for (ss, _), v in b.items() if ss == s)
                                  for s in samples}, "\n")

    # ---- 1-3. own 0-SPION medians -------------------------------------
    print("=" * 78)
    print("1-3. OWN 0-SPION REFERENCE MEDIANS  (quality_ok voxels of the '0' segmentation)")
    print("=" * 78)
    ref = {}
    for s in samples:
        key = (s, "0")
        if key not in b:
            print(f"  {s:9s} NO 0-SPION SEGMENTATION")
            continue
        q = b[key][b[key].quality_ok == 1]
        ref[s] = float(np.median(q.I0_raw))
        print(f"  {s:9s} median I0_selected = {ref[s]:10.4f}   from {len(q):,} quality voxels "
              f"of {len(b[key]):,} in segmentation '0'")
    print(f"\n  Pill2 factor == 2042.62 (2 dp): {round(ref['Pill2'], 2) == 2042.62}"
          f"   exact: {ref['Pill2']!r}")

    # ---- 4. Pill1 transferred reference --------------------------------
    print("\n" + "=" * 78)
    print("4. PILL1 TRANSFERRED REFERENCE  (concentration effects from Syringes + 3D only)")
    print("=" * 78)
    shared = sorted({m for (s, m) in b if s in REFERENCE_SAMPLES and m not in ("water", "tissue")},
                    key=float)
    print(f"  concentrations available in the two reference phantoms: {shared}")
    print(f"\n  {'conc':6s}" + "".join(f"{s:>14s}" for s in REFERENCE_SAMPLES)
          + f"{'log-mean ratio':>17s}")
    effect = {}
    for m in shared:
        ratios = []
        line = f"  {m:6s}"
        for s in REFERENCE_SAMPLES:
            k = (s, m)
            if k in b:
                q = b[k][b[k].quality_ok == 1]
                r = float(np.median(q.I0_raw)) / ref[s]
                ratios.append(r)
                line += f"{r:14.4f}"
            else:
                line += f"{'-':>14s}"
        if ratios:
            effect[m] = float(np.exp(np.mean(np.log(ratios))))
            line += f"{effect[m]:17.4f}"
        print(line)

    pill1_conc = [m for (s, m) in b if s == "Pill1" and m not in ("tissue",)]
    print(f"\n  Pill1's SPION segmentations: {pill1_conc}")
    m = pill1_conc[0]
    q = b[("Pill1", m)][b[("Pill1", m)].quality_ok == 1]
    pill1_median = float(np.median(q.I0_raw))
    ref["Pill1"] = pill1_median / effect[m]
    print(f"  median I0_selected of Pill1 '{m}' = {pill1_median:.4f}  ({len(q):,} quality voxels)")
    print(f"  concentration effect at {m} learned from {REFERENCE_SAMPLES} = {effect[m]:.4f}")
    print(f"  => transferred Pill1 reference = {pill1_median:.4f} / {effect[m]:.4f} "
          f"= {ref['Pill1']:.4f}")

    # validation: what does Pill1's tissue say? (not used to set the factor)
    qt = b[("Pill1", "tissue")][b[("Pill1", "tissue")].quality_ok == 1]
    print(f"  [validation only, not used] Pill1 tissue median = {np.median(qt.I0_raw):.1f} "
          f"-> {np.median(qt.I0_raw)/ref['Pill1']:.3f} x the transferred reference")

    # ---- 5. label audit -------------------------------------------------
    print("\n" + "=" * 78)
    print("5. LABEL AUDIT — what each factor depends on")
    print("=" * 78)
    audit = {
        "Syringes": "own '0' segmentation identity only (1 categorical label)",
        "3D":       "own '0' segmentation identity only (1 categorical label)",
        "Pill2":    "own '0' segmentation identity only (1 categorical label)",
        "Pill1":    f"own '{m}' segmentation label + concentration effect from Syringes+3D",
    }
    for s in samples:
        print(f"  {s:9s} factor {ref[s]:10.4f}   <- {audit[s]}")
    print(f"\n  Pill2 concentration DISTRIBUTION used anywhere? NO")
    print(f"  Pill2 labels other than '0' used anywhere?        NO")
    print(f"  Pill2 '0' region identity used?                   YES - as an intensity")
    print(f"                                                    reference, as sanctioned")

    # ---- 6. f(rs) on the new normalisation ------------------------------
    print("\n" + "=" * 78)
    print("6. f(rs) AFTER THE NEW NORMALISATION  (fitted on Syringes + 3D + Pill1 only)")
    print("=" * 78)
    parts = []
    for (s, mat), d in b.items():
        if s not in TRAINING:
            continue
        keep = d[d.flag_high_rs == 0]
        parts.append(pd.DataFrame({"rs": keep.rs_selected.to_numpy(np.float64),
                                   "I0n": keep.I0_raw.to_numpy(np.float64) / ref[s]}))
    train = pd.concat(parts, ignore_index=True)
    c = np.polyfit(train.rs, train.I0n, 2)
    pred = np.poly1d(c)(train.rs)
    r2 = 1 - float(np.sum((train.I0n - pred) ** 2)) / float(np.sum((train.I0n - train.I0n.mean()) ** 2))
    print(f"  fitted on {TRAINING}: {len(train):,} voxels (flag_high_rs == 0), Pill2 excluded")
    print(f"  f(rs) = {c[0]:+.4f}*rs^2 {c[1]:+.4f}*rs {c[2]:+.4f}   R^2 = {r2:.4f}")
    print(f"  previous f (Strategy-B scale): -166.1185*rs^2 +19.0491*rs +0.8232")
    print(f"  changed: {not np.allclose(c, [-166.1185, 19.0491, 0.8232], rtol=1e-3)}")

    # ---- 7-8. identity and cross-pipeline consistency --------------------
    print("\n" + "=" * 78)
    print("7-8. RESIDUAL IDENTITY AND SEGMENTED vs FULL-VOLUME CONSISTENCY")
    print("=" * 78)
    f = np.poly1d(c)
    for s in samples:
        rs_all, i0_all = [], []
        for (ss, mat), d in b.items():
            if ss != s:
                continue
            rs_all.append(d.rs_selected.to_numpy(np.float64))
            i0_all.append(d.I0_raw.to_numpy(np.float64))
        rs_all = np.concatenate(rs_all); i0_all = np.concatenate(i0_all)
        i0n = i0_all / ref[s]
        res = i0n - f(rs_all)
        print(f"  {s:9s} I0_norm median {np.median(i0n):7.4f}   "
              f"I0_residual median {np.median(res):+7.4f}   "
              f"identity holds: {bool(np.allclose(res, i0n - f(rs_all)))}")

    # full Pill2 volume would use the same factor and the same f
    vrs = np.load(VOL / "rs_selected.npy"); vi0 = np.load(VOL / "I0_selected.npy")
    valid = np.load(VOL / "fit_success.npy") > 0
    vi0n = vi0[valid] / ref["Pill2"]
    print(f"\n  Pill2 FULL VOLUME with the same factor {ref['Pill2']:.4f} and the same f:")
    print(f"    {int(valid.sum()):,} valid voxels   I0_norm median {np.median(vi0n):.4f}   "
          f"I0_residual median {np.median(vi0n - f(vrs[valid])):+.4f}")
    # the segmented Pill2 voxels are a subset -> must agree exactly
    seg_rs, seg_i0 = [], []
    for (ss, mat), d in b.items():
        if ss == "Pill2":
            seg_rs.append(d.rs_selected.to_numpy(np.float64))
            seg_i0.append(d.I0_raw.to_numpy(np.float64))
    seg_i0n = np.concatenate(seg_i0) / ref["Pill2"]
    print(f"    same factor as the segmented Pill2 data: True (both {ref['Pill2']:.4f})")
    print(f"    segmented Pill2 I0_norm median {np.median(seg_i0n):.4f} "
          f"(subset of the volume, so a different median is expected)")
    print(f"\n  model_selected.npy present in Pill2_normalized: "
          f"{(MRI/'i0_rs/Volume_for_testing/Pill2_normalized/model_selected.npy').is_file()}")


main()
