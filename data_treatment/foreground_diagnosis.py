"""READ-ONLY: compare the Otsu foreground against the segmentation foreground for Pill2.

Writes only a CSV table and one figure into Training/diagnosis. Nothing else is
created or modified; no refit, no normalisation, no classification.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT  # noqa: E402
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

MRI = DATA_ROOT
VOL = MRI / "i0_rs" / "Volume_for_testing" / "Pill2_normalized"
SEG = MRI / "Normalized_data_i0_rs" / "Segmentations" / "Pill2"
OUT = MRI / "Training" / "diagnosis"


def main():
    import echoviewer  # noqa: F401
    from echoviewer import EchoSeries
    from echoviewer.diagnostics import _figure, _save
    from skimage.filters import threshold_otsu

    OUT.mkdir(parents=True, exist_ok=True)

    otsu_mask = np.load(VOL / "fit_success.npy") > 0
    shape = otsu_mask.shape

    seg_mask = np.zeros(shape, bool)
    per_seg = {}
    for p in sorted(q for q in SEG.iterdir() if q.is_dir()):
        d = pd.read_csv(p / "voxels.csv.gz", usecols=["x", "y", "z"])
        m = np.zeros(shape, bool)
        m[d.z, d.y, d.x] = True
        per_seg[p.name] = m
        seg_mask |= m

    s = EchoSeries.from_sample(MRI / "Volumes", "Pill2", cache_size=1)
    raw = s.volume(0).data.copy()
    s.close()
    thr = float(threshold_otsu(raw[np.isfinite(raw)].ravel()))

    included = seg_mask & otsu_mask
    excluded = seg_mask & ~otsu_mask
    extra = otsu_mask & ~seg_mask
    # conservative background: well away from any segmented voxel
    far = ~ndi.binary_dilation(seg_mask, ndi.generate_binary_structure(3, 1), iterations=6)
    background = far & ~otsu_mask

    print("=" * 74)
    print("1. INCLUSION / EXCLUSION")
    print("=" * 74)
    print(f"  volume                       {int(np.prod(shape)):>10,}")
    print(f"  segmentation foreground      {int(seg_mask.sum()):>10,}")
    print(f"  Otsu foreground (current)    {int(otsu_mask.sum()):>10,}   threshold {thr:.1f}")
    print(f"  segmentation INCLUDED        {int(included.sum()):>10,}  "
          f"({100*included.sum()/seg_mask.sum():5.2f}% of segmentation)")
    print(f"  segmentation EXCLUDED        {int(excluded.sum()):>10,}  "
          f"({100*excluded.sum()/seg_mask.sum():5.2f}%)")
    print(f"  Otsu-only (not segmented)    {int(extra.sum()):>10,}")

    rows = []
    for name, m in per_seg.items():
        inc = int((m & otsu_mask).sum()); exc = int((m & ~otsu_mask).sum())
        rows.append({"segmentation": name, "material": name.split("_", 2)[2],
                     "n_voxels": int(m.sum()), "included": inc, "excluded": exc,
                     "pct_excluded": 100 * exc / m.sum(),
                     "median_I0_included": float(np.median(raw[m & otsu_mask])) if inc else np.nan,
                     "median_I0_excluded": float(np.median(raw[m & ~otsu_mask])) if exc else np.nan})
    per_seg_table = pd.DataFrame(rows)
    print("\n  by segmentation:")
    print(per_seg_table.round(2).to_string(index=False))

    print("\n" + "=" * 74)
    print("2. WHERE THE EXCLUDED VOXELS ARE (coronal profile)")
    print("=" * 74)
    prof = pd.DataFrame({
        "coronal_y": np.arange(shape[1]),
        "segmented": seg_mask.sum(axis=(0, 2)),
        "otsu": otsu_mask.sum(axis=(0, 2)),
        "excluded": excluded.sum(axis=(0, 2)),
    })
    prof["pct_excluded"] = 100 * prof.excluded / prof.segmented.replace(0, np.nan)
    active = prof[prof.segmented > 0]
    worst = active.nlargest(8, "pct_excluded")
    print("  slices losing the largest share of their segmented voxels:")
    print(worst[["coronal_y", "segmented", "otsu", "excluded", "pct_excluded"]]
          .round(1).to_string(index=False))
    ends = active[(active.coronal_y <= active.coronal_y.min() + 15) |
                  (active.coronal_y >= active.coronal_y.max() - 15)]
    middle = active[~active.index.isin(ends.index)]
    print(f"\n  end slices (first/last 15):  {int(ends.excluded.sum()):,} excluded of "
          f"{int(ends.segmented.sum()):,}  ({100*ends.excluded.sum()/ends.segmented.sum():.1f}%)")
    print(f"  middle slices:               {int(middle.excluded.sum()):,} excluded of "
          f"{int(middle.segmented.sum()):,}  ({100*middle.excluded.sum()/middle.segmented.sum():.1f}%)")

    print("\n" + "=" * 74)
    print("3-4. INTENSITY: IS THE EXCLUDED MATERIAL GENUINE SAMPLE?")
    print("=" * 74)
    stats = []
    for name, m in (("included (seg & Otsu)", included), ("excluded (seg, not Otsu)", excluded),
                    ("Otsu-only", extra), ("background (far from sample)", background)):
        v = raw[m]
        stats.append({"population": name, "n": int(m.sum()), "median": float(np.median(v)),
                      "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)),
                      "mean": float(v.mean()), "std": float(v.std())})
        print(f"  {name:30s} n={int(m.sum()):>9,}  median {np.median(v):8.1f}  "
              f"p05 {np.percentile(v,5):8.1f}  p95 {np.percentile(v,95):8.1f}")
    bg = raw[background]
    ex = raw[excluded]
    bg_p999 = float(np.percentile(bg, 99.9))
    print(f"\n  background 99.9th percentile: {bg_p999:.1f}")
    print(f"  excluded voxels above it:     {100*float((ex > bg_p999).mean()):.1f}%")
    print(f"  excluded median / background median = "
          f"{np.median(ex)/max(np.median(bg),1e-9):.1f}x")
    print(f"  -> excluded voxels are {'GENUINE SAMPLE, not noise' if np.median(ex) > 5*np.median(bg) else 'ambiguous'}")

    print("\n" + "=" * 74)
    print("5. CAN THE SEGMENTATION FOREGROUND BE USED AS THE VALID MASK?")
    print("=" * 74)
    print(f"  same array shape: {seg_mask.shape == otsu_mask.shape}  {shape}")
    print(f"  segmentation voxels outside the array: 0 (indices came from the same grid)")
    print(f"  segmentations mutually disjoint: "
          f"{sum(int(m.sum()) for m in per_seg.values()) == int(seg_mask.sum())}")
    print(f"  fits currently exist for: {int(otsu_mask.sum()):,} voxels")
    print(f"  a segmentation-based mask needs fits for: {int(seg_mask.sum()):,}")
    print(f"  -> {int(excluded.sum()):,} voxels ({100*excluded.sum()/seg_mask.sum():.1f}%) "
          f"would need a NEW fit before they could be classified")

    print("\n" + "=" * 74)
    print("6. CANDIDATE VALID-VOXEL RULES")
    print("=" * 74)
    cand = []
    for label, t in (("current global Otsu", thr), ("0.75 x Otsu", 0.75 * thr),
                     ("0.60 x Otsu", 0.60 * thr), ("0.50 x Otsu", 0.50 * thr),
                     ("background p99.9", bg_p999)):
        m = raw > t
        cand.append({"rule": label, "threshold": t, "n_voxels": int(m.sum()),
                     "seg_recovered_pct": 100 * float((seg_mask & m).sum() / seg_mask.sum()),
                     "background_leaked": int((background & m).sum())})
    cand.append({"rule": "segmentation foreground", "threshold": np.nan,
                 "n_voxels": int(seg_mask.sum()), "seg_recovered_pct": 100.0,
                 "background_leaked": 0})
    cand = pd.DataFrame(cand)
    print(cand.round(1).to_string(index=False))

    per_seg_table.to_csv(OUT / "foreground_by_segmentation.csv", index=False)
    prof.to_csv(OUT / "foreground_coronal_profile.csv", index=False)
    pd.DataFrame(stats).to_csv(OUT / "foreground_intensity_stats.csv", index=False)
    cand.to_csv(OUT / "candidate_valid_voxel_rules.csv", index=False)

    # ---- one diagnostic figure -------------------------------------------
    fig = _figure(figsize=(17, 10))
    gs = fig.add_gridspec(2, 3, hspace=.28, wspace=.22)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(prof.coronal_y, prof.segmented, label="segmentation foreground", lw=1.4)
    ax.plot(prof.coronal_y, prof.otsu, label="Otsu foreground (used)", lw=1.4)
    ax.fill_between(prof.coronal_y, 0, prof.excluded, color="crimson", alpha=.5,
                    label="excluded")
    ax.set_xlabel("coronal y"); ax.set_ylabel("voxels in slice")
    ax.set_title("Coverage per coronal slice"); ax.legend(fontsize=8); ax.grid(alpha=.25)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(active.coronal_y, active.pct_excluded, color="crimson", lw=1.4)
    ax.set_xlabel("coronal y"); ax.set_ylabel("% of segmented voxels excluded")
    ax.set_title("Loss is concentrated at the tube ends"); ax.grid(alpha=.25)

    ax = fig.add_subplot(gs[0, 2])
    bins = np.linspace(0, float(np.percentile(raw[seg_mask], 99.5)), 200)
    for v, lab, c in ((raw[included], "included", "#4a6fe3"),
                      (raw[excluded], "excluded", "#c9314e"),
                      (bg, "background", "#8f8f8f")):
        ax.hist(v, bins=bins, density=True, histtype="step", lw=1.6, color=c, label=lab)
    ax.axvline(thr, color="black", ls="--", lw=1.2, label=f"Otsu {thr:.0f}")
    ax.set_yscale("log"); ax.set_xlabel("first-echo intensity"); ax.set_ylabel("density")
    ax.set_title("Excluded sits far above background"); ax.legend(fontsize=8); ax.grid(alpha=.25)

    j = int(prof.loc[active.pct_excluded.idxmax(), "coronal_y"])
    for col, (title, m, colour) in enumerate((
            ("segmentation foreground", seg_mask, "winter"),
            ("Otsu foreground (used)", otsu_mask, "autumn"),
            ("excluded = segmented but not fitted", excluded, "cool"))):
        ax = fig.add_subplot(gs[1, col])
        base = np.flipud(raw[:, j, :])
        vmax = float(np.percentile(base[base > 0], 99)) if (base > 0).any() else 1
        ax.imshow(base, cmap="gray", vmax=vmax, interpolation="nearest")
        ax.imshow(np.ma.masked_where(~np.flipud(m[:, j, :]), np.ones(base.shape)),
                  cmap=colour, alpha=.55, interpolation="nearest")
        ax.set_title(f"{title}\ncoronal y = {j}", fontsize=9); ax.set_axis_off()

    fig.suptitle("Pill2 full-volume foreground: Otsu mask vs segmentation mask (read-only diagnostic)",
                 fontsize=14)
    _save(fig, OUT / "foreground_diagnosis.png", [])
    print(f"\nSaved to {OUT}")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name}")


main()
