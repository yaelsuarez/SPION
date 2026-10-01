"""Same material across phantoms: overlap before vs after Strategy-B normalisation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT  # noqa: E402
import numpy as np, pandas as pd
from itertools import combinations

MRI = DATA_ROOT
DATA = MRI / "Normalized_data_i0_rs" / "Segmentations"
OUT = MRI / "i0_rs" / "model_diagnostics" / "normalized_feature_space"
ORDER = ("0", "0.05", "0.1", "0.2", "0.25", "0.3", "water", "tissue")
SAMPLES = ("Syringes", "3D", "Pill1", "Pill2")

def main():
    import echoviewer  # noqa: F401
    from echoviewer.diagnostics import bhattacharyya_overlap, _figure, _save

    blocks = {}
    for sd in sorted(DATA.iterdir()):
        if not sd.is_dir(): continue
        for gd in sorted(sd.iterdir()):
            if not gd.is_dir(): continue
            df = pd.read_csv(gd/"voxels.csv.gz",
                             usecols=["rs_selected","I0_raw","I0_normalized","flag_high_rs"])
            df = df[df.flag_high_rs == 0]
            blocks[(sd.name, gd.name.split("_",2)[2])] = df.reset_index(drop=True)

    def feat(key, mode):
        d = blocks[key]
        col = {"raw":"I0_raw","norm":"I0_normalized"}[mode]
        return np.column_stack([d.rs_selected.to_numpy(np.float64), d[col].to_numpy(np.float64)])
    def rs_only(key):
        return blocks[key].rs_selected.to_numpy(np.float64)[:,None]

    rows = []
    for material in ORDER:
        present = [s for s in SAMPLES if (s, material) in blocks]
        for a, b in combinations(present, 2):
            ka, kb = (a, material), (b, material)
            rows.append({
                "material": material, "sample_A": a, "sample_B": b,
                "n_A": len(blocks[ka]), "n_B": len(blocks[kb]),
                "rs_only": bhattacharyya_overlap(rs_only(ka), rs_only(kb)),
                "raw_rs_I0": bhattacharyya_overlap(feat(ka,"raw"), feat(kb,"raw")),
                "norm_rs_I0": bhattacharyya_overlap(feat(ka,"norm"), feat(kb,"norm")),
            })
    t = pd.DataFrame(rows)
    t["change"] = t.norm_rs_I0 - t.raw_rs_I0

    print("SAME MATERIAL, DIFFERENT PHANTOM — overlap in (rs, I0). 1.0 = phantoms agree.")
    print("high-rs voxels excluded; complete remaining population\n")
    print(f"{'material':9s}{'sample A':10s}{'sample B':10s}{'n_A':>9s}{'n_B':>9s}"
          f"{'rs only':>9s}{'raw I0':>9s}{'norm I0':>9s}{'change':>9s}")
    print("-"*83)
    for _, r in t.iterrows():
        print(f"{r.material:9s}{r.sample_A:10s}{r.sample_B:10s}{r.n_A:>9,}{r.n_B:>9,}"
              f"{r.rs_only:>9.3f}{r.raw_rs_I0:>9.3f}{r.norm_rs_I0:>9.3f}{r.change:>+9.3f}")
    print("-"*83)
    print(f"{'MEDIAN':29s}{'':18s}{t.rs_only.median():>9.3f}"
          f"{t.raw_rs_I0.median():>9.3f}{t.norm_rs_I0.median():>9.3f}{t.change.median():>+9.3f}")
    print(f"{'MEAN':29s}{'':18s}{t.rs_only.mean():>9.3f}"
          f"{t.raw_rs_I0.mean():>9.3f}{t.norm_rs_I0.mean():>9.3f}{t.change.mean():>+9.3f}")
    print(f"\nimproved by normalisation: {int((t.change>0).sum())}/{len(t)} pairs")

    t.to_csv(OUT/"overlap_same_material_across_phantoms.csv", index=False)

    fig = _figure(figsize=(12,6)); ax = fig.subplots()
    x = np.arange(len(t)); w = 0.27
    ax.bar(x-w, t.rs_only, w, label="rs only", color="#8f8f8f")
    ax.bar(x,   t.raw_rs_I0, w, label="rs + raw I0", color="#c9314e")
    ax.bar(x+w, t.norm_rs_I0, w, label="rs + normalised I0", color="#4a6fe3")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.material}\n{r.sample_A[:4]}–{r.sample_B[:4]}" for r in t.itertuples()],
                       fontsize=7)
    ax.set_ylabel("overlap (1.0 = phantoms agree)")
    ax.set_title("Same material in different phantoms — higher is better here.\n"
                 "Strategy-B normalisation, high-rs voxels excluded, complete population")
    ax.axhline(1.0, color="black", ls="--", lw=0.8); ax.legend(); ax.grid(alpha=.25, axis="y")
    _save(fig, OUT/"overlap_same_material_across_phantoms.png", [])
    print(f"\nSaved: {OUT/'overlap_same_material_across_phantoms.csv'}")
    print(f"       {OUT/'overlap_same_material_across_phantoms.png'}")

main()
