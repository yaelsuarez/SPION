"""Shape-agnostic segmentation, adapted from 2D slices to whole 3D volumes.

Ported from the ``relaxivity_calc_mac.ipynb`` "General (shape-agnostic)
segmentation" cell, which detects samples of arbitrary shape rather than
assuming circles or donuts. That notebook is left untouched; this is a
parallel implementation.

The notebook's pipeline, and what each step becomes here:

===========================  ====================================================
Notebook (per 2D slice)      This module (whole volume)
===========================  ====================================================
``get_gray``: ``img*255/max``  normalised **once over the entire volume**
``binarize_image``           :func:`binarize_volume` - Otsu on the 3D histogram,
                             a fixed level, or a 3D Gaussian local mean
``clean_binary``             :func:`clean_binary_volume` - opening/closing with a
                             3D ball, 3D hole filling
``split_touching_regions``   :func:`split_touching_3d` - watershed on the 3D
                             distance transform
``split_by_edges``           :func:`split_by_edges_3d` - watershed on the 3D
                             gradient magnitude
``cv2.connectedComponents``  :func:`scipy.ndimage.label` with 26-connectivity
``min_area``/``max_area``    ``min_voxels``/``max_fraction`` (volumes, not areas)
``erode_px``                 ``erode_voxels``, applied per region in 3D
===========================  ====================================================

Why not just run the 2D code on every slice: the notebook rescales each image
by *its own* maximum, so slice 40 and slice 41 get different intensity scales
and therefore different thresholds. Run independently, neighbouring slices
disagree about where an object ends, and the "same" object picks up a
different label on each slice. Normalising and thresholding across the whole
volume, then labelling with 3D connectivity, is what makes one physical object
come out as one region with one index through the entire stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import apply_hysteresis_threshold, threshold_multiotsu, threshold_otsu
from skimage.morphology import ball
from skimage.segmentation import watershed

from dicomview.volume import StackAxis, Volume

#: Label value reserved for background.
BACKGROUND = 0


@dataclass(frozen=True)
class SegmentationParams:
    """Tuning for :func:`segment_volume`, mirroring the notebook's arguments.

    Attributes:
        method: ``"otsu"`` (global, automatic), ``"fixed"`` (level ``C``) or
            ``"adaptive"`` (local mean minus ``C``).
        C: Threshold level for ``"fixed"``; offset below the local mean for
            ``"adaptive"``. Both are on the 0-255 normalised scale.
        block_size: Neighbourhood size for ``"adaptive"``, in voxels.
        blur: Standard deviation of the 3D Gaussian pre-blur, in voxels. The
            notebook passed an odd kernel size to ``cv2.GaussianBlur``; a sigma
            is the natural equivalent for ``scipy.ndimage``.
        open_size: Radius of the 3D ball used for morphological opening
            (removes speckle). 0 disables.
        close_size: Radius of the 3D ball used for closing (bridges gaps).
        fill_holes: Fill interior cavities. Left off by default so that
            donut-shaped samples stay hollow, as in the notebook.
        min_voxels: Regions smaller than this are dropped.
        max_fraction: Regions larger than this fraction of the volume are
            dropped, which discards a background-sized blob.
        split_edges: Cut regions that touch but differ in intensity, using a
            watershed on the gradient. Needed when one object holds several
            bands or compartments.
        grad_pct: Percentile of the gradient used to pick flat interior seeds.
        seed_min: Seeds smaller than this are noise and are ignored.
        split_touching: Cut regions that touch at a neck, using a watershed on
            the distance transform. Ignored when ``split_edges`` is set.
        fg_ratio: Fraction of the peak distance that counts as "sure
            foreground" for ``split_touching``.
        erode_voxels: Shell of voxels removed from every region, to drop
            partial-volume boundaries.
        sort_by: ``"position"`` (through the stack, then top-to-bottom,
            then left-to-right), ``"volume"`` (largest first) or ``None``.
    """

    method: str = "otsu"
    C: float = 0.0
    block_size: int = 21
    blur: float = 0.8
    open_size: int = 1
    close_size: int = 1
    fill_holes: bool = False
    min_voxels: int = 200
    max_fraction: float = 0.9
    split_edges: bool = False
    grad_pct: float = 45.0
    seed_min: int = 50
    split_touching: bool = False
    fg_ratio: float = 0.5
    erode_voxels: int = 1
    sort_by: str | None = "position"
    hysteresis_ratio: float = 0.6
    split_bands: bool = True
    max_bands: int = 4
    band_contrast: float = 8.0
    band_min_share: float = 0.05
    band_coverage: float = 0.95
    band_seed_min: int = 200
    band_sharpness: float = 0.5


#: Named presets. The first two carry over the notebook's two parameter sets;
#: ``AUTO_PRESET`` picks between them from the data.
PRESETS: dict[str, SegmentationParams] = {
    "Small samples (syringes)": SegmentationParams(
        method="adaptive", block_size=21, C=2.0, blur=0.8,
        open_size=1, close_size=1, min_voxels=200, erode_voxels=1,
    ),
    "Large objects with bands (pill / 3D print)": SegmentationParams(
        method="otsu", blur=0.8, open_size=1, close_size=1,
        min_voxels=500, erode_voxels=1,
        split_edges=True, grad_pct=45.0, seed_min=200,
    ),
    "Plain threshold (no splitting)": SegmentationParams(
        method="otsu", blur=0.8, open_size=1, close_size=1,
        min_voxels=200, erode_voxels=1,
    ),
}

#: Label used in the UI for the data-driven choice.
AUTO_PRESET = "Auto"

#: A single region covering more than this fraction of the foreground, and more
#: than :data:`_AUTO_VOLUME_SHARE` of the whole volume, means one object has
#: swallowed everything and needs the edge watershed to be split into bands.
_AUTO_FOREGROUND_SHARE = 0.6
_AUTO_VOLUME_SHARE = 0.02

#: The edge watershed is kept only if it cuts the largest region's share of the
#: foreground to below this multiple of what plain labelling gave. A
#: homogeneous sample has no intensity steps to cut on, so the split changes
#: almost nothing and is discarded rather than shedding noise fragments.
_AUTO_SPLIT_ACCEPT = 0.85


@dataclass(frozen=True)
class Segment3D:
    """One connected 3D region.

    Attributes:
        index: 1-based label value inside :attr:`Segmentation.labels`.
        voxels: Region size in voxels.
        centroid: ``(k, j, i)`` centre of mass in volume index space.
        bbox: ``(k0, k1, j0, j1, i0, i1)`` half-open bounding box.
    """

    index: int
    voxels: int
    centroid: tuple[float, float, float]
    bbox: tuple[int, int, int, int, int, int]

    @property
    def name(self) -> str:
        """Label shown in the UI, e.g. ``"Segmentation 3"``."""
        return f"Segmentation {self.index}"


@dataclass(frozen=True)
class Segmentation:
    """The result of segmenting one volume: a label map plus per-region metadata.

    Regions are stored as a single integer label volume rather than one boolean
    array per region. A boolean mask per region would cost ~5 MB each for this
    dataset; the label volume costs ~11 MB in total, and a cross-section is one
    array slice plus a comparison.

    Attributes:
        labels: Integer volume, ``0`` for background and ``1..n`` for regions.
        segments: Metadata for each region, ordered as displayed.
        params: The parameters that produced this result.
        source_shape: Shape of the volume that was segmented. Used to refuse
            applying these masks to a volume of a different size.
        preset: Name of the preset used, for display.
    """

    labels: np.ndarray
    segments: tuple[Segment3D, ...]
    params: SegmentationParams
    source_shape: tuple[int, int, int]
    preset: str = ""
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __len__(self) -> int:
        return len(self.segments)

    def matches(self, volume: Volume) -> bool:
        """True if these masks can be laid over ``volume`` voxel for voxel."""
        return tuple(volume.shape) == tuple(self.source_shape)

    def as_volume(self) -> Volume:
        """Wrap the label map so it can be sliced by :meth:`Volume.extract`.

        Reusing the viewer's own plane extraction is what guarantees the
        overlay lines up with the greyscale image in all three planes: the
        axis order and the vertical flip applied to coronal and sagittal
        planes come from exactly one implementation, not two.
        """
        return Volume(
            data=self.labels,
            spacing=(1.0, 1.0, 1.0),
            stack_axis=StackAxis.SPATIAL,
            description="segmentation labels",
        )

    def mask(self, index: int) -> np.ndarray:
        """Boolean mask of one region, built on demand from the label map."""
        return self.labels == index


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def to_gray_volume(data: np.ndarray) -> np.ndarray:
    """Scale a volume to 0-255 floats using a single global maximum.

    The notebook's ``get_gray`` divides each 2D image by its own maximum. Doing
    that per slice gives every slice a different intensity scale, so a fixed or
    Otsu threshold means something different on each one. Normalising over the
    whole volume keeps one scale for the entire stack, which is what lets a
    single threshold describe one physical object from top to bottom.
    """
    data = np.asarray(data, dtype=np.float32)
    finite = data[np.isfinite(data)]
    peak = float(finite.max()) if finite.size else 0.0
    if peak <= 0:
        return np.zeros_like(data, dtype=np.float32)
    gray = np.nan_to_num(data, nan=0.0, posinf=peak, neginf=0.0) * (255.0 / peak)
    return np.clip(gray, 0.0, 255.0, out=gray)


def binarize_volume(gray: np.ndarray, params: SegmentationParams) -> np.ndarray:
    """Threshold a normalised volume into a boolean foreground mask.

    Args:
        gray: Volume scaled to 0-255 by :func:`to_gray_volume`.
        params: Which method to use and its settings.

    Returns:
        Boolean array, True for foreground.

    Raises:
        ValueError: If ``params.method`` is not recognised.
    """
    working = ndi.gaussian_filter(gray, sigma=params.blur) if params.blur > 0 else gray

    if params.method == "otsu":
        # Otsu over the whole 3D histogram, so one level covers every slice.
        if float(working.max()) <= float(working.min()):
            return np.zeros(working.shape, dtype=bool)
        # Flattened: the threshold is the same, and it skips skimage's
        # "looks like an RGB image" shape heuristic on small volumes.
        return working > threshold_otsu(working.ravel())

    if params.method == "fixed":
        return working > params.C

    if params.method == "adaptive":
        # 3D counterpart of cv2's ADAPTIVE_THRESH_GAUSSIAN_C: compare each
        # voxel with a Gaussian-weighted local mean of its neighbourhood.
        sigma = max(params.block_size / 6.0, 0.5)
        local_mean = ndi.gaussian_filter(working, sigma=sigma)
        return working > (local_mean + params.C)

    raise ValueError(f"Unknown threshold method: {params.method!r}")


def clean_binary_volume(binary: np.ndarray, params: SegmentationParams) -> np.ndarray:
    """Remove speckle and bridge small gaps, in 3D.

    Args:
        binary: Boolean foreground mask.
        params: Opening/closing radii and hole filling.

    Returns:
        The cleaned boolean mask.
    """
    pad = max(params.open_size or 0, params.close_size or 0)
    if pad <= 0:
        return ndi.binary_fill_holes(binary) if params.fill_holes else binary

    # Pad by repeating the edge before any morphology. Both opening and closing
    # end in an erosion, which would otherwise trim the first and last slices of
    # every object that reaches the edge of the field of view - a sample that
    # simply continues past the imaged volume, not a real boundary.
    padded = np.pad(binary, pad, mode="edge")
    if params.open_size and params.open_size > 0:
        padded = ndi.binary_opening(padded, structure=ball(params.open_size))
    if params.close_size and params.close_size > 0:
        padded = ndi.binary_closing(padded, structure=ball(params.close_size))
    binary = padded[tuple(slice(pad, -pad) for _ in range(padded.ndim))]

    if params.fill_holes:
        binary = ndi.binary_fill_holes(binary)
    return binary


def split_touching_3d(binary: np.ndarray, params: SegmentationParams) -> np.ndarray:
    """Cut regions joined at a narrow neck, via a watershed on the 3D distance map.

    The 3D counterpart of the notebook's ``split_touching_regions``.

    Returns:
        Integer label volume, 0 for background.
    """
    if not binary.any():
        return np.zeros(binary.shape, dtype=np.int32)

    distance = ndi.distance_transform_edt(binary)
    peak = float(distance.max())
    if peak <= 0:
        return np.zeros(binary.shape, dtype=np.int32)

    markers, _ = ndi.label(distance > params.fg_ratio * peak)
    if markers.max() == 0:
        return np.zeros(binary.shape, dtype=np.int32)
    return watershed(-distance, markers, mask=binary).astype(np.int32)


def split_by_edges_3d(
    gray: np.ndarray, binary: np.ndarray, params: SegmentationParams
) -> np.ndarray:
    """Cut touching regions that differ in intensity, via a watershed on edges.

    The 3D counterpart of the notebook's ``split_by_edges``. A distance-based
    split only cuts where the shape narrows, so it cannot separate bands
    stacked inside one straight object. Here the seeds are the flat,
    low-gradient interiors and the watershed lines settle on the intensity
    steps between them, which also survives a shading gradient along the
    object.

    Returns:
        Integer label volume, 0 for background.
    """
    if not binary.any():
        return np.zeros(binary.shape, dtype=np.int32)

    smoothed = ndi.gaussian_filter(gray, sigma=max(params.blur, 0.5))
    gradient = np.sqrt(
        sum(ndi.sobel(smoothed, axis=axis).astype(np.float32) ** 2 for axis in range(3))
    )

    cutoff = np.percentile(gradient[binary], params.grad_pct)
    seeds = (gradient <= cutoff) & binary
    seed_labels, n_seeds = ndi.label(seeds)
    if n_seeds == 0:
        return np.zeros(binary.shape, dtype=np.int32)

    # Drop seeds made of noise, then renumber the survivors from 1.
    sizes = np.bincount(seed_labels.ravel())
    keep = [label for label in range(1, n_seeds + 1) if sizes[label] >= params.seed_min]
    if not keep:
        return np.zeros(binary.shape, dtype=np.int32)

    markers = np.zeros(binary.shape, dtype=np.int32)
    for new_label, old_label in enumerate(keep, start=1):
        markers[seed_labels == old_label] = new_label

    return watershed(gradient, markers, mask=binary).astype(np.int32)


def _gradient_magnitude(gray: np.ndarray, blur: float) -> np.ndarray:
    """3D Sobel gradient magnitude, used as the watershed relief."""
    smoothed = ndi.gaussian_filter(gray, sigma=max(blur, 0.5))
    return np.sqrt(
        sum(ndi.sobel(smoothed, axis=axis).astype(np.float32) ** 2 for axis in range(3))
    )


def grow_into_dim_material(
    gray: np.ndarray, labels: np.ndarray, params: SegmentationParams
) -> np.ndarray:
    """Expand each labelled region into connected dimmer material.

    A compartment can be much darker than its neighbours and still be part of
    the same object: the last band of the capsule in Pill2 sits below the Otsu
    level entirely and is lost without this, so the capsule comes out one
    compartment short.

    Plain hysteresis thresholding recovers it, but lowering the threshold
    globally also bridges structures that were properly separate - it fuses the
    wedges of the ``3D`` phantom to the ring around them, collapsing six
    regions into three. So the growth is *seeded*: the regions found at the
    high threshold are the markers, and a watershed expands them into the
    low-threshold mask. Every region gains its own dim material and no two
    regions can merge, because a watershed never merges its markers.

    Args:
        gray: Normalised volume.
        labels: Regions found at the high threshold.
        params: ``hysteresis_ratio`` sets the low threshold, as a fraction of
            the Otsu level. 0 disables the step.

    Returns:
        The grown label volume, or ``labels`` unchanged if disabled.
    """
    if not params.hysteresis_ratio or not 0 < params.hysteresis_ratio < 1:
        return labels
    if labels.max() == 0:
        return labels

    blurred = ndi.gaussian_filter(gray, sigma=params.blur) if params.blur > 0 else gray
    level = threshold_otsu(blurred.ravel())
    dim = apply_hysteresis_threshold(blurred, level * params.hysteresis_ratio, level)
    mask = clean_binary_volume(dim, params) | (labels > 0)
    return watershed(_gradient_magnitude(gray, params.blur), labels, mask=mask)


def _band_candidate(
    gray: np.ndarray,
    gradient: np.ndarray,
    region: np.ndarray,
    classes: int,
    params: SegmentationParams,
) -> tuple[np.ndarray, list[float]] | None:
    """Try to cut one region into ``classes`` intensity compartments.

    Seeds are the eroded cores of the region's intensity classes, and the
    watershed runs on the gradient so the boundaries follow the real intensity
    steps rather than the flat class thresholds.

    Returns:
        ``(labels_within_region, part_means)`` or None if fewer than two
        usable seeds were found.
    """
    values = gray[region]
    if values.size == 0:
        return None
    try:
        # nbins is kept modest deliberately: multi-Otsu searches the histogram
        # exhaustively, so its cost grows as nbins**(classes-1). At 256 bins a
        # four-class split takes seconds per region; at 64 it is milliseconds,
        # and the thresholds land in the same place on data quantised to 0-255.
        thresholds = threshold_multiotsu(values, classes=classes, nbins=64)
    except ValueError:
        return None  # Not enough distinct levels for this many classes.

    class_map = np.where(region, np.digitize(gray, thresholds), -1)
    seeds = np.zeros(region.shape, dtype=np.int32)
    next_seed = 1
    for class_index in range(classes):
        core = ndi.binary_erosion(
            class_map == class_index, structure=ndi.generate_binary_structure(3, 1)
        )
        cores, n_cores = ndi.label(core)
        sizes = np.bincount(cores.ravel())
        for core_index in range(1, n_cores + 1):
            if sizes[core_index] >= params.band_seed_min:
                seeds[cores == core_index] = next_seed
                next_seed += 1
    if next_seed <= 2:
        return None

    labels = watershed(gradient, seeds, mask=region)
    means = [
        float(gray[labels == value].mean())
        for value in range(1, next_seed)
        if (labels == value).any()
    ]
    return labels, means


def _step_sharpness(
    gray: np.ndarray, labels: np.ndarray, keep: list[int], reach: int = 2
) -> float:
    """How much of the difference between parts survives at their interface.

    For each pair of touching parts this compares the mean intensity in the
    thin shells either side of their shared surface with the difference between
    the parts as a whole::

        sharpness = |mean(A near B) - mean(B near A)| / |mean(A) - mean(B)|

    A real boundary between compartments is a step: the full difference is
    already there within a voxel or two of the interface, so the ratio
    approaches 1. A smooth intensity ramp along an object has no boundary at
    all - values just either side of any cut through it are nearly equal - so
    the ratio collapses towards 0, however different the two halves look on
    average.

    This is what tells the capsule in Pill2 (0.74) apart from the syringes
    (0.13-0.36), whose brightness falls off gradually along their length.
    Mean difference alone cannot: the syringes' parts differ by 25-29 grey
    levels, more than the capsule's 15.

    Returns:
        The worst ratio over all touching pairs, or 0.0 if none touch.
    """
    structure = ndi.generate_binary_structure(3, 1)
    worst = np.inf
    for first in range(len(keep)):
        for second in range(first + 1, len(keep)):
            part_a = labels == keep[first]
            part_b = labels == keep[second]
            near_a = part_a & ndi.binary_dilation(part_b, structure, reach)
            near_b = part_b & ndi.binary_dilation(part_a, structure, reach)
            if not near_a.any() or not near_b.any():
                continue  # These two parts do not touch.
            overall = abs(float(gray[part_a].mean()) - float(gray[part_b].mean()))
            if overall < 1e-6:
                return 0.0
            local = abs(float(gray[near_a].mean()) - float(gray[near_b].mean()))
            worst = min(worst, local / overall)
    return 0.0 if not np.isfinite(worst) else float(worst)


def _accept_bands(
    gray: np.ndarray,
    labels: np.ndarray,
    means: list[float],
    region_size: int,
    params: SegmentationParams,
) -> list[int] | None:
    """Decide whether a candidate split describes real compartments.

    Three things have to hold, and together they separate genuine compartments
    from a homogeneous region being cut up arbitrarily:

    * every part is a substantial share of the region - a real compartment is
      not a sliver shaved off the edge;
    * the parts between them cover essentially the whole region;
    * consecutive parts differ in mean intensity by at least
      ``band_contrast``. This is the decisive one. The capsule in Pill2 splits
      into parts averaging 105, 87, 69 and 48, each ~18 grey levels apart. The
      surrounding tube wall, asked for the same treatment, returns parts
      averaging 66.2 and 65.4 - a gap of 0.8, which is the signature of a cut
      through uniform material rather than a boundary between compartments.

    Returns:
        The accepted label values, or None to keep the region whole.
    """
    values = [v for v in range(1, len(means) + 1)]
    sizes = [int((labels == v).sum()) for v in values]
    kept = [
        (value, size, mean)
        for value, size, mean in zip(values, sizes, means)
        if size >= max(params.min_voxels, params.band_min_share * region_size)
    ]
    if len(kept) < 2:
        return None
    if sum(size for _, size, _ in kept) < params.band_coverage * region_size:
        return None

    ordered = sorted(mean for _, _, mean in kept)
    gaps = np.diff(ordered)
    if gaps.size == 0 or float(gaps.min()) < params.band_contrast:
        return None

    values = [value for value, _, _ in kept]
    if _step_sharpness(gray, labels, values) < params.band_sharpness:
        return None
    return values


def split_regions_into_bands(
    gray: np.ndarray, labels: np.ndarray, params: SegmentationParams
) -> np.ndarray:
    """Split each region into intensity compartments, region by region.

    This is the 3D counterpart of the notebook's ``split_by_edges``, with one
    change that matters: the notebook derives its seed threshold from a
    percentile of the gradient over the *whole* foreground. In a volume holding
    both a small sample and a large homogeneous holder, the holder dominates
    that statistic and the sample's internal steps never register. In Pill2 the
    capsule's own gradient sits at 23 while the foreground percentile is 190,
    so the compartments inside it were invisible and the whole capsule came out
    as one mask.

    Working per region puts every threshold on the scale of the object being
    cut, which is what lets compartments that sit close together be told apart.
    Regions that hold no compartments are left exactly as they were.

    Args:
        gray: Normalised volume.
        labels: Result of the connected-components pass.
        params: Tuning; see the ``band_*`` fields.

    Returns:
        A new label volume. Regions that were split take new label values;
        untouched regions keep their voxels.
    """
    if not params.split_bands or labels.max() == 0:
        return labels

    gradient = _gradient_magnitude(gray, params.blur)
    sizes = np.bincount(labels.ravel())
    windows = ndi.find_objects(labels)
    out = np.zeros_like(labels)
    next_label = 1

    for region_value, window in enumerate(windows, start=1):
        region_size = int(sizes[region_value]) if region_value < sizes.size else 0
        if window is None or region_size == 0:
            continue

        # Everything below works on the region's bounding box. Running the
        # watershed over the whole volume for every region and every candidate
        # class count is minutes of work; cropped it is a fraction of a second.
        sub_region = labels[window] == region_value
        sub_gray = gray[window]
        sub_gradient = gradient[window]

        accepted: tuple[np.ndarray, list[int]] | None = None
        # Too small to hold compartments worth separating: leave it alone.
        if region_size >= max(4 * params.min_voxels, 2 * params.band_seed_min):
            # Prefer the finest split that still passes: the capsule genuinely
            # holds four bands, and stopping at two would merge three of them.
            for classes in range(2, params.max_bands + 1):
                candidate = _band_candidate(
                    sub_gray, sub_gradient, sub_region, classes, params
                )
                if candidate is None:
                    continue
                part_labels, means = candidate
                keep = _accept_bands(
                    sub_gray, part_labels, means, region_size, params
                )
                if keep is not None and (accepted is None or len(keep) > len(accepted[1])):
                    accepted = (part_labels, keep)

        target = out[window]
        if accepted is None:
            target[sub_region] = next_label
            next_label += 1
            continue

        part_labels, keep = accepted
        claimed = np.zeros(sub_region.shape, dtype=np.int32)
        for value in keep:
            claimed[(part_labels == value) & sub_region] = next_label
            next_label += 1

        # Voxels the accepted parts did not claim go to their nearest part
        # rather than dropping out of the segmentation.
        orphan = sub_region & (claimed == 0)
        if orphan.any():
            _, nearest = ndi.distance_transform_edt(claimed == 0, return_indices=True)
            claimed[orphan] = claimed[tuple(idx[orphan] for idx in nearest)]
        target[sub_region] = claimed[sub_region]

    return out


def _erode_regions(labels: np.ndarray, radius: int) -> np.ndarray:
    """Shrink every region by ``radius`` voxels, independently of its neighbours.

    Eroding the whole foreground at once would leave voxels on the shared face
    of two touching regions. Eroding each region on its own removes the
    partial-volume shell all the way round, which is what the notebook's
    per-mask ``erode_px`` does in 2D.
    """
    if radius <= 0:
        return labels

    structure = ball(radius)
    out = np.zeros_like(labels)
    objects = ndi.find_objects(labels)
    for label_value, window in enumerate(objects, start=1):
        if window is None:
            continue
        # Pad the window by the radius so erosion sees true background outside.
        padded = tuple(
            slice(max(sl.start - radius, 0), min(sl.stop + radius, size))
            for sl, size in zip(window, labels.shape)
        )
        region = labels[padded] == label_value
        # border_value=1 treats the edge of the field of view as inside the
        # region. Erosion is meant to drop partial-volume voxels at a
        # sample/background interface; the face where a sample simply runs out
        # of the imaged volume is not such an interface, and eroding there
        # would chew slices off the ends of every object that reaches the edge.
        eroded = ndi.binary_erosion(region, structure=structure, border_value=1)
        if not eroded.any():
            eroded = region  # Never erode a region out of existence.
        out[padded][eroded] = label_value
    return out


def _describe_regions(labels: np.ndarray) -> list[Segment3D]:
    """Collect voxel count, centroid and bounding box for every label."""
    n_labels = int(labels.max())
    if n_labels == 0:
        return []

    counts = np.bincount(labels.ravel(), minlength=n_labels + 1)
    # Ask only about labels that still exist. The size filter leaves gaps in the
    # numbering, and center_of_mass divides by the label's mass, so querying a
    # vanished label would return NaN and warn.
    indices = [label for label in range(1, n_labels + 1) if counts[label] > 0]
    if not indices:
        return []
    centroids = ndi.center_of_mass(labels > 0, labels, indices)
    boxes = ndi.find_objects(labels)

    segments: list[Segment3D] = []
    for label_value, centroid in zip(indices, centroids):
        window = boxes[label_value - 1]
        if window is None:
            continue
        segments.append(
            Segment3D(
                index=label_value,
                voxels=int(counts[label_value]),
                centroid=tuple(float(c) for c in centroid),  # type: ignore[arg-type]
                bbox=(
                    window[0].start, window[0].stop,
                    window[1].start, window[1].stop,
                    window[2].start, window[2].stop,
                ),
            )
        )
    return segments


def _sort_segments(segments: list[Segment3D], sort_by: str | None) -> list[Segment3D]:
    """Order regions for display, mirroring the notebook's ``sort_by``."""
    if sort_by == "position":
        # Band the two slower axes so that near-equal positions read in rows,
        # exactly as the notebook's round(y/10) does in 2D.
        return sorted(
            segments,
            key=lambda s: (round(s.centroid[0] / 10), round(s.centroid[1] / 10), s.centroid[2]),
        )
    if sort_by == "volume":
        return sorted(segments, key=lambda s: s.voxels, reverse=True)
    return segments


def _renumber(labels: np.ndarray, segments: Iterable[Segment3D]) -> tuple[np.ndarray, tuple[Segment3D, ...]]:
    """Renumber the label volume to 1..n following the display order."""
    lookup = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    ordered: list[Segment3D] = []
    for new_index, segment in enumerate(segments, start=1):
        lookup[segment.index] = new_index
        ordered.append(replace(segment, index=new_index))
    return lookup[labels], tuple(ordered)


def _dominance(labels: np.ndarray, binary: np.ndarray) -> float:
    """Fraction of the foreground taken by the single largest region."""
    foreground = int(binary.sum())
    if foreground == 0 or labels.max() == 0:
        return 0.0
    sizes = np.bincount(labels.ravel())[1:]
    return float(sizes.max()) / foreground if sizes.size else 0.0


def _needs_edge_split(binary: np.ndarray, labels: np.ndarray) -> bool:
    """Decide whether one region has swallowed the whole object.

    Used by the ``Auto`` preset. Cheap: it only looks at the region sizes that
    have already been computed, so choosing a strategy never costs a second
    pass over the data.
    """
    foreground = int(binary.sum())
    if foreground == 0 or labels.max() == 0:
        return False
    sizes = np.bincount(labels.ravel())[1:]
    if sizes.size == 0:
        return False
    largest = int(sizes.max())
    return (
        largest / foreground > _AUTO_FOREGROUND_SHARE
        and largest / binary.size > _AUTO_VOLUME_SHARE
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def segment_volume(
    volume: Volume,
    params: SegmentationParams | None = None,
    preset: str = AUTO_PRESET,
) -> Segmentation:
    """Segment a whole volume into coherent 3D regions.

    Threshold -> 3D clean-up -> optional watershed split -> 3D connected
    components -> size filter -> per-region erosion. Interior cavities stay
    background unless ``fill_holes`` is set, so donut-shaped samples come out
    hollow without a special case, as in the notebook.

    Args:
        volume: The volume to segment; only :attr:`Volume.data` is used.
        params: Tuning. Defaults to the preset named by ``preset``.
        preset: Preset name, used for display and to select ``params`` when it
            is not given. :data:`AUTO_PRESET` chooses between splitting and not
            splitting from the size of the largest region.

    Returns:
        A :class:`Segmentation`. Empty (``len() == 0``) when nothing survives
        the size filters, which is a valid outcome, not an error.
    """
    auto = params is None and preset == AUTO_PRESET
    if params is None:
        params = PRESETS.get(preset, PRESETS["Plain threshold (no splitting)"])

    gray = to_gray_volume(volume.data)

    if auto:
        # Evaluate the notebook's two thresholding strategies and keep the
        # better one. This happens once, when the sample is opened; the result
        # is cached and reused for every echo time.
        candidates = [
            _segment_with(volume, gray, PRESETS["Plain threshold (no splitting)"], True),
            _segment_with(volume, gray, PRESETS["Small samples (syringes)"], False),
        ]
        chosen = min(candidates, key=lambda c: _size_spread(c))
        return replace(chosen, preset=AUTO_PRESET, warnings=chosen.warnings + (
            f"Auto chose the {'local (adaptive)' if chosen.params.method == 'adaptive' else 'global (Otsu)'}"
            " threshold as the more even split of this sample.",
        ))

    return _segment_with(volume, gray, params, allow_auto_split=False, preset=preset)


def _size_spread(segmentation: Segmentation) -> float:
    """Coefficient of variation of region sizes; lower is a more even split.

    These phantoms hold a set of comparable samples - syringes at different
    concentrations, wedges of one printed part. A result whose regions come out
    similarly sized is the one that captured each physical sample once, rather
    than clipping the dim ones or shedding slivers off a solid object. An empty
    result sorts last so it is never preferred.
    """
    sizes = np.array([s.voxels for s in segmentation.segments], dtype=float)
    if sizes.size == 0:
        return float("inf")
    if sizes.size == 1:
        return 1.0
    return float(sizes.std() / sizes.mean())


def _segment_with(
    volume: Volume,
    gray: np.ndarray,
    params: SegmentationParams,
    allow_auto_split: bool,
    preset: str = "",
) -> Segmentation:
    """Run the pipeline once with one set of parameters."""
    binary = clean_binary_volume(binarize_volume(gray, params), params)

    warnings: list[str] = []
    if not binary.any():
        return Segmentation(
            labels=np.zeros(volume.shape, dtype=np.int32),
            segments=(),
            params=params,
            source_shape=tuple(volume.shape),
            preset=preset,
            warnings=("Thresholding found no foreground; try a different preset.",),
        )

    plain_labels, _ = ndi.label(binary, structure=np.ones((3, 3, 3)))

    if params.split_edges:
        labels = split_by_edges_3d(gray, binary, params)
    elif params.split_touching:
        labels = split_touching_3d(binary, params)
    elif allow_auto_split and _needs_edge_split(binary, plain_labels):
        # One region has swallowed the object. Try the edge watershed, but keep
        # it only if it genuinely breaks the object up: on a homogeneous sample
        # there are no intensity steps to cut on, and forcing a split there
        # just sheds noise fragments off a region that was already correct.
        split = split_by_edges_3d(gray, binary, replace(params, seed_min=200))
        if _dominance(split, binary) < _AUTO_SPLIT_ACCEPT * _dominance(plain_labels, binary):
            labels = split
            params = replace(params, split_edges=True, seed_min=200)
            warnings.append(
                "One region covered most of the object, so the edge watershed "
                "was used to split it into bands."
            )
        else:
            labels = plain_labels
    else:
        labels = plain_labels

    # Recover dim material belonging to a region before looking for
    # compartments, so a band that sits below the threshold is not simply
    # missing from the split.
    if params.method in ("otsu", "fixed"):
        labels = grow_into_dim_material(gray, labels, params)

    # Compartments inside a single connected object, decided per region so that
    # a large holder cannot hide the steps inside a small sample.
    if params.split_bands:
        labels = split_regions_into_bands(gray, labels, params)

    # Size filter, then per-region erosion, then display order.
    max_voxels = params.max_fraction * labels.size
    sizes = np.bincount(labels.ravel())
    drop = {
        label
        for label in range(1, int(labels.max()) + 1)
        if not (params.min_voxels <= sizes[label] <= max_voxels)
    }
    if drop:
        keep_lookup = np.arange(int(labels.max()) + 1, dtype=np.int32)
        for label in drop:
            keep_lookup[label] = BACKGROUND
        labels = keep_lookup[labels]

    labels = _erode_regions(labels, params.erode_voxels)

    # Filter again: erosion shrinks every region, so one that was just above
    # min_voxels before can fall below it after, leaving a speck in the list.
    if params.erode_voxels > 0:
        sizes = np.bincount(labels.ravel())
        too_small = [
            label
            for label in range(1, int(labels.max()) + 1)
            if sizes[label] and sizes[label] < params.min_voxels
        ]
        if too_small:
            lookup = np.arange(int(labels.max()) + 1, dtype=np.int32)
            for label in too_small:
                lookup[label] = BACKGROUND
            labels = lookup[labels]

    segments = _sort_segments(_describe_regions(labels), params.sort_by)
    labels, segments = _renumber(labels, segments)

    if not segments:
        warnings.append(
            "No region passed the size filters; lower min_voxels or change preset."
        )

    return Segmentation(
        labels=labels.astype(np.int32),
        segments=segments,
        params=params,
        source_shape=tuple(volume.shape),
        preset=preset,
        warnings=tuple(warnings),
    )
