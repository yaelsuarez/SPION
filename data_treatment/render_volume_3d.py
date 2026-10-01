#!/usr/bin/env python3
"""Isosurface rendering of one echo-time volume, shown from several viewpoints.

Read-only. The volume is downsampled and smoothed for the surface extraction
only; nothing on disk is touched.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_ROOT  # noqa: E402

import numpy as np  # noqa: E402

SAMPLE = DATA_ROOT / "Volumes" / "Pill1"
ECHO_MS = 208.0
SPACING = 0.32          # mm, isotropic
STEP = 1                # downsample factor before marching cubes (1 = full res)
VIEWS = [(18, 30)]
DPI = 600               # print resolution; only bites on raster output


def main(out_path) -> int:
    from scipy.ndimage import gaussian_filter
    from skimage.filters import threshold_otsu
    from skimage.measure import marching_cubes
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from echoviewer.dataset import get_available_echotimes, load_volume_from_folder
    from echoviewer.diagnostics import _figure

    started = time.time()
    echoes = get_available_echotimes(SAMPLE)
    echo = min(echoes, key=lambda e: abs(e.value - ECHO_MS))
    volume = load_volume_from_folder(echo.path).data.astype(np.float32)
    print(f"{SAMPLE.name}  TE {echo.value:g} ms  shape {volume.shape}")

    # The threshold still comes from a smoothed, downsampled copy: Otsu on the
    # raw voxels is pulled around by noise. The surface itself is extracted from
    # the untouched volume, so no detail is blurred away.
    level = float(threshold_otsu(gaussian_filter(volume[::2, ::2, ::2], sigma=1.0)))
    small = volume[::STEP, ::STEP, ::STEP]
    spacing = (SPACING * STEP,) * 3
    verts, faces, normals, _ = marching_cubes(small, level=level, spacing=spacing,
                                              step_size=1)
    print(f"  isosurface at {level:.0f}: {len(verts):,} vertices, {len(faces):,} faces")

    # simple lambertian shading from the vertex normals, averaged per face
    light = np.array([0.45, 0.35, 0.82])
    light = light / np.linalg.norm(light)
    shade = np.clip(normals[faces].mean(axis=1) @ light, 0, 1)
    from matplotlib import colormaps
    # grey ramp: unlit faces stay near black, fully lit faces go to white
    colours = colormaps["gray"](0.12 + 0.88 * shade)

    fig = _figure(figsize=(3.75 * len(VIEWS), 4.6))
    fig.patch.set_facecolor("black")
    for i, (elev, azim) in enumerate(VIEWS, start=1):
        ax = fig.add_subplot(1, len(VIEWS), i, projection="3d")
        ax.set_facecolor("black")
        mesh = Poly3DCollection(verts[faces], facecolors=colours, linewidths=0)
        ax.add_collection3d(mesh)
        ax.set_xlim(0, verts[:, 0].max()); ax.set_ylim(0, verts[:, 1].max())
        ax.set_zlim(0, verts[:, 2].max())
        ax.set_box_aspect([verts[:, k].max() for k in range(3)])
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(f"elev {elev}°, azim {azim}°", fontsize=9, color="white")
    fig.suptitle(f"Pill1 — isosurface of the {echo.value:g} ms echo "
                 f"(Otsu level {level:.0f}), 0.32 mm isotropic", fontsize=13,
                 color="white")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="black")
    fig.clf()
    print(f"  saved {out_path}\n  elapsed {time.time()-started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "pill1_3d.png"))
