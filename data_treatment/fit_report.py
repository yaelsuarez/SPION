"""Summarise the finished relaxometry run from the saved maps."""
import json
from pathlib import Path

import numpy as np
from config import DATA_ROOT  # noqa: E402

R = DATA_ROOT / "i0_rs"


def stats(values):
    v = values[np.isfinite(values)]
    return (v.mean(), np.median(v), v.std(), np.percentile(v, 5), np.percentile(v, 95)) if v.size else (0,) * 5


def walk(root):
    for meta in sorted(root.rglob("metadata.json")):
        yield meta.parent


total_vox = total_ok = 0
total_A = total_B = 0
print(f"{'dataset':44s} {'voxels':>9s} {'fail%':>6s} {'rs_B med':>9s} {'I0_B med':>9s} {'B med':>8s} {'dR2 med':>8s} {'A/B':>13s}")
print("-" * 118)
for source in ("Segmentations", "Volumes"):
    for d in walk(R / source):
        md = json.loads((d / "metadata.json").read_text())
        succ = np.load(d / "fit_success_mask.npy") > 0
        n = int(md["n_voxels_evaluated"])
        ok = int(succ.sum())
        choice = np.load(d / "model_choice_map.npy")
        a = int((choice == 1).sum())
        b = int((choice == 2).sum())
        rsB = np.load(d / "rs_map_B.npy")[succ]
        i0B = np.load(d / "I0_map_B.npy")[succ]
        Bm = np.load(d / "B_map.npy")[succ]
        dR2 = np.load(d / "delta_R2_map.npy")[succ]
        label = f"{source[:3]} {md.get('sample_name','')}/{md.get('segmentation_name','(volume)')}"
        print(f"{label:44s} {n:9,d} {100*(1-ok/n):6.2f} {np.median(rsB):9.5f} {np.median(i0B):9.1f} "
              f"{np.median(Bm):8.1f} {np.median(dR2):8.4f} {a:6,d}/{b:6,d}")
        total_vox += n
        total_ok += ok
        total_A += a
        total_B += b

print("-" * 118)
print(f"TOTAL voxels fitted: {total_vox:,}   successful: {total_ok:,} ({100*total_ok/total_vox:.2f}%)   "
      f"failures: {total_vox-total_ok:,} ({100*(1-total_ok/total_vox):.3f}%)")
print(f"Model A preferred: {total_A:,} ({100*total_A/total_vox:.1f}%)   "
      f"Model B preferred: {total_B:,} ({100*total_B/total_vox:.1f}%)")

# echo times: identical everywhere?
ets = {tuple(json.loads((d / 'metadata.json').read_text())['echo_times']) for d in walk(R)}
print(f"\nDistinct echo-time sets across all datasets: {len(ets)}")
for e in ets:
    print(f"  {len(e)} echoes: {e[0]:g}..{e[-1]:g} ms, step {e[1]-e[0]:g}")

# global parameter distributions over segmentations only
print("\nParameter distributions (segmentations, successful voxels):")
for name, fn in (("I0_noB", "I0_map_noB"), ("rs_noB", "rs_map_noB"),
                 ("I0_B", "I0_map_B"), ("rs_B", "rs_map_B"), ("B", "B_map")):
    pool = []
    for d in walk(R / "Segmentations"):
        s = np.load(d / "fit_success_mask.npy") > 0
        pool.append(np.load(d / f"{fn}.npy")[s])
    v = np.concatenate(pool)
    m, md_, sd, p5, p95 = stats(v)
    print(f"  {name:7s} mean={m:10.4f} median={md_:10.4f} std={sd:10.4f} p05={p5:10.4f} p95={p95:10.4f}")

size = sum(f.stat().st_size for f in R.rglob("*") if f.is_file())
print(f"\nOutput size: {size/1e9:.2f} GB under {R}")
