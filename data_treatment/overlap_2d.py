"""2D (rs_selected, I0_normalized) overlap: by material, and by sample within material."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT  # noqa: E402
import numpy as np, pandas as pd

MRI = DATA_ROOT
DATA = MRI / "Normalized_data_i0_rs" / "Segmentations"
OUT = MRI / "i0_rs" / "model_diagnostics" / "normalized_feature_space"
LABEL_ORDER = ("0", "0.05", "0.1", "0.2", "0.25", "0.3", "water", "tissue")
SAMPLES = ("Syringes", "3D", "Pill1", "Pill2")


def main():
    import echoviewer  # noqa: F401
    from echoviewer.diagnostics import _figure, _save, bhattacharyya_overlap

    # load, keeping (sample, material) blocks
    blocks = {}
    for sample_dir in sorted(DATA.iterdir()):
        if not sample_dir.is_dir():
            continue
        for seg_dir in sorted(sample_dir.iterdir()):
            if not seg_dir.is_dir():
                continue
            df = pd.read_csv(seg_dir / "voxels.csv.gz",
                             usecols=["rs_selected", "I0_normalized", "flag_high_rs"])
            # Take the material from the folder name, not the CSV column: pandas
            # infers "0" as int64 in small files and float64 in large ones, which
            # silently split 3D's 301,646 zero-SPION voxels into a phantom "0.0"
            # class of their own.
            key = (sample_dir.name, seg_dir.name.split("_", 2)[2])
            blocks.setdefault(key, []).append(df)
            del df
    blocks = {k: pd.concat(v, ignore_index=True) for k, v in blocks.items()}

    def features(sample=None, material=None, drop_high_rs=True):
        parts = []
        for (s, m), df in blocks.items():
            if (sample and s != sample) or (material and m != material):
                continue
            block = df[df["flag_high_rs"] == 0] if drop_high_rs else df
            parts.append(np.column_stack([block["rs_selected"].to_numpy(np.float64),
                                          block["I0_normalized"].to_numpy(np.float64)]))
        return np.vstack(parts) if parts else None

    saved = []
    materials = [m for m in LABEL_ORDER if any(k[1] == m for k in blocks)]

    # ---- A. material x material, pooled over samples -------------------
    matrix = np.full((len(materials), len(materials)), np.nan)
    counts = {}
    for i, a in enumerate(materials):
        fa = features(material=a)
        counts[a] = len(fa)
        for j, b in enumerate(materials):
            matrix[i, j] = 1.0 if i == j else bhattacharyya_overlap(fa, features(material=b))

    print("=" * 78)
    print("A. OVERLAP BY MATERIAL in (rs_selected, I0_normalized), pooled over samples")
    print("   high-rs voxels excluded; complete remaining population")
    print("=" * 78)
    print(f"{'':9s}" + "".join(f"{m:>8s}" for m in materials))
    for i, a in enumerate(materials):
        print(f"{a:9s}" + "".join(f"{matrix[i, j]:8.3f}" for j in range(len(materials))))
    print("\nn voxels: " + ", ".join(f"{m}={counts[m]:,}" for m in materials))

    # ---- B. sample x sample, within each material ----------------------
    print("\n" + "=" * 78)
    print("B. OVERLAP BY SAMPLE within each material  (1.0 = phantoms agree)")
    print("=" * 78)
    rows = []
    for material in materials:
        present = [s for s in SAMPLES if (s, material) in blocks]
        if len(present) < 2:
            print(f"\n{material:8s} only in {present[0] if present else 'none'} — no comparison")
            continue
        print(f"\n{material}:")
        print(f"{'':10s}" + "".join(f"{s:>10s}" for s in present))
        for a in present:
            fa = features(sample=a, material=material)
            line = f"{a:10s}"
            for b in present:
                value = 1.0 if a == b else bhattacharyya_overlap(
                    fa, features(sample=b, material=material))
                line += f"{value:10.3f}"
                if a < b:
                    rows.append({"material": material, "sample_a": a, "sample_b": b,
                                 "overlap": value})
            print(line)
    cross = pd.DataFrame(rows)

    # ---- figure ---------------------------------------------------------
    figure = _figure(figsize=(16, 6))
    left, right = figure.subplots(1, 2, width_ratios=[1, 1.15])
    image = left.imshow(matrix, cmap="RdYlGn_r", vmin=0, vmax=1)
    left.set_xticks(range(len(materials)));  left.set_xticklabels(materials, rotation=45, ha="right")
    left.set_yticks(range(len(materials)));  left.set_yticklabels(materials)
    for i in range(len(materials)):
        for j in range(len(materials)):
            left.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7,
                      color="white" if matrix[i, j] > 0.6 else "black")
    left.set_title("Between materials (pooled over samples)")
    figure.colorbar(image, ax=left, fraction=0.046)

    if len(cross):
        order = [m for m in materials if m in set(cross["material"])]
        pairs = sorted({(r.sample_a, r.sample_b) for r in cross.itertuples()})
        grid = np.full((len(order), len(pairs)), np.nan)
        for r in cross.itertuples():
            grid[order.index(r.material), pairs.index((r.sample_a, r.sample_b))] = r.overlap
        image = right.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1)
        right.set_xticks(range(len(pairs)))
        right.set_xticklabels([f"{a}\nvs {b}" for a, b in pairs], fontsize=7)
        right.set_yticks(range(len(order)));  right.set_yticklabels(order)
        for i in range(len(order)):
            for j in range(len(pairs)):
                if np.isfinite(grid[i, j]):
                    right.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=7,
                               color="black" if grid[i, j] > 0.35 else "white")
        right.set_title("Same material, different phantom\n(green = phantoms agree)")
        figure.colorbar(image, ax=right, fraction=0.046)
    figure.suptitle("Bhattacharyya overlap in (rs_selected, I0_normalized), Strategy-B "
                    "normalisation, high-rs voxels excluded", fontsize=12)
    _save(figure, OUT / "overlap_rs_I0_normalized.png", saved)

    pd.DataFrame(matrix, index=materials, columns=materials).to_csv(
        OUT / "overlap_by_material.csv")
    cross.to_csv(OUT / "overlap_by_sample_within_material.csv", index=False)
    print(f"\nSaved: {OUT/'overlap_rs_I0_normalized.png'}")
    print(f"       overlap_by_material.csv, overlap_by_sample_within_material.csv")


if __name__ == "__main__":
    main()
