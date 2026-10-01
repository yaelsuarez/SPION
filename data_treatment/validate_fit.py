"""Validation run for the voxel-wise fitting, on one segmentation and one volume."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT, REPO  # noqa: E402
import numpy as np


def main():
    import echoviewer  # noqa: F401  (bootstraps dicomview)
    from echoviewer.relaxometry import (find_segmentations, find_echo_folders,
                                        parse_segmentation_name, _load_npz_volume,
                                        fit_segmentation, _foreground_mask)
    from echoviewer.fitting import fit_one_voxel, model_no_baseline, model_with_baseline

    S = DATA_ROOT / "Segmentations"
    OUT = REPO / "outputs" / "fittest"
    OUT.mkdir(parents=True, exist_ok=True)

    print("=== 1. segmentation names / labels preserved ===")
    for n in ["Segmentation_6_0.2", "Segmentation_4_water", "Segmentation_2_tissue", "Segmentation_2_0"]:
        print(f"   {n:26s} -> number={parse_segmentation_name(n)[0]}  label={parse_segmentation_name(n)[1]!r}")

    segs = find_segmentations(S / "Pill1")
    print(f"\n=== 2. Pill1 segmentations found: {[d.name for d in segs]} ===")
    d = segs[1]
    echoes = find_echo_folders(d)
    times = [t for t, _ in echoes]
    print(f"   {d.name}: {len(echoes)} echo folders")
    print(f"   echo times (numeric, sorted): {times[:8]} ... {times[-2:]}")
    print(f"   evenly spaced? {bool(np.allclose(np.diff(times), np.diff(times)[0]))}  (not assumed either way)")

    v0 = _load_npz_volume(echoes[0][1])
    mask = np.load(S / "Pill1" / "volumes" / d.name / "mask.npz")["mask"]
    print(f"\n=== 3. .npz spatial shape preserved: {v0.shape} ===")
    print(f"   nonzero voxels = {int((v0 != 0).sum()):,}   saved mask = {int(mask.sum()):,}")
    print(f"   nonzero == saved mask: {bool(np.array_equal(v0 != 0, mask))}")

    # 4. a single voxel, fitted by hand, to confirm the plumbing
    coords = np.nonzero(v0 != 0)
    pick = int(np.argmax(v0[coords]))
    zyx = (coords[0][pick], coords[1][pick], coords[2][pick])
    curve = np.array([_load_npz_volume(f)[zyx] for _, f in echoes], dtype=float)
    r = fit_one_voxel(np.array(times, dtype=float), curve)
    print(f"\n=== 4. one voxel at (z,y,x)={zyx} ===")
    print(f"   intensities: {np.round(curve[:6], 1)} ... {np.round(curve[-2:], 1)}")
    print(f"   Model A: I0={r['I0_noB']:.1f} rs={r['rs_noB']:.5f} /ms (T2={1/r['rs_noB']:.1f} ms) R2={r['R2_noB']:.4f}")
    print(f"   Model B: I0={r['I0_B']:.1f} rs={r['rs_B']:.5f} /ms B={r['B']:.1f} R2={r['R2_B']:.4f}")
    print(f"   delta_R2={r['delta_R2']:+.4f}  model_selected={int(r['model_selected'])} (1=A, 2=B, by AIC)")

    print("\n=== 5. full segmentation fit ===")
    t0 = time.time()
    rep = fit_segmentation(d, OUT / "Pill1" / d.name, "Pill1", workers=9)
    print(f"   {rep.summary}   ({time.time() - t0:.1f}s)")
    for k in ("I0_noB", "rs_noB", "I0_B", "rs_B", "B", "R2_noB", "R2_B", "delta_R2"):
        s = rep.stats[k]
        print(f"   {k:9s} median={s['median']:11.4f}  p05={s['p05']:11.4f}  p95={s['p95']:11.4f}")

    print("\n=== 6. outputs ===")
    outdir = OUT / "Pill1" / d.name
    for f in sorted(outdir.iterdir()):
        if f.suffix == ".npy":
            a = np.load(f)
            nz = int((a != 0).sum())
            print(f"   {f.name:22s} shape={a.shape} dtype={str(a.dtype):8s} nonzero={nz:,}")
        else:
            print(f"   {f.name:22s} {f.stat().st_size/1e6:.2f} MB")

    print("\n=== 7. spatial correspondence ===")
    i0 = np.load(outdir / "I0_map_noB.npy")
    succ = np.load(outdir / "fit_success_mask.npy")
    print(f"   map shape == volume shape: {i0.shape == v0.shape}")
    print(f"   fitted voxels all inside the segmentation: {bool(((succ > 0) & (v0 == 0)).sum() == 0)}")
    print(f"   background left at zero: {bool((i0[v0 == 0] == 0).all())}")
    import pandas as pd
    csv = pd.read_csv(outdir / "voxel_fit_data.csv")
    print(f"   CSV rows {len(csv):,} == evaluated voxels {int((v0 != 0).sum()):,}: {len(csv) == int((v0 != 0).sum())}")
    row = csv.iloc[0]
    zz, yy, xx = int(row.z), int(row.y), int(row.x)
    print(f"   CSV row 0 (z,y,x)=({zz},{yy},{xx}) I0_noB={row.I0_noB:.3f} == map {i0[zz,yy,xx]:.3f}: "
          f"{np.isclose(row.I0_noB, i0[zz,yy,xx], rtol=1e-5)}")

    print("\n=== 8. complete-volume foreground rule ===")
    from echoviewer import EchoSeries
    V = str(DATA_ROOT / "Volumes")
    s = EchoSeries.from_sample(V, "Pill1", cache_size=1)
    vol = s.volume(0)
    fg = _foreground_mask(vol.data)
    print(f"   Pill1 volume {vol.shape}: {int(fg.sum()):,} voxels to fit "
          f"({100*fg.mean():.1f}% of {vol.data.size:,})")
    s.close()


if __name__ == "__main__":
    main()
