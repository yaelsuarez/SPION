"""Command-line entry point: ``python -m echoviewer <Volumes> [--sample NAME]``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .dataset import get_available_echotimes, get_available_samples
from .series import DEFAULT_CACHE_SIZE

from dicomview.loader import DicomLoadError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="echoviewer",
        description="Browse DICOM volumes across slices and echo times.",
    )
    parser.add_argument("volumes", help="The Volumes directory holding the samples.")
    parser.add_argument("--sample", help="Sample to open first, e.g. Syringes.")
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the samples and echo times found, then exit.",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=DEFAULT_CACHE_SIZE,
        help=f"Volumes kept in memory (default: {DEFAULT_CACHE_SIZE}; ~22 MB each).",
    )
    parser.add_argument(
        "--export-segmentations",
        metavar="OUT_DIR",
        help="Headless: segment each sample once and write the sparse and dense "
        "exports under OUT_DIR, then exit.",
    )
    parser.add_argument(
        "--only",
        metavar="SAMPLE",
        nargs="+",
        help="Restrict --export-segmentations to these samples.",
    )
    parser.add_argument(
        "--sparse-only",
        action="store_true",
        help="With --export-segmentations, skip the dense volumes.",
    )
    parser.add_argument(
        "--dense-only",
        action="store_true",
        help="With --export-segmentations, skip the sparse tables.",
    )
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="With --export-segmentations, write only the preview images "
        "(re-renders previews without rewriting the data).",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="With --export-segmentations, skip the preview images.",
    )
    parser.add_argument(
        "--fit-relaxometry",
        metavar="OUT_DIR",
        help="Headless: fit both decay models voxel-wise and write the maps, "
        "CSVs and metadata under OUT_DIR/Segmentations and OUT_DIR/Volumes.",
    )
    parser.add_argument(
        "--segmentations",
        metavar="SEGMENTATIONS_DIR",
        help="Where the saved segmentation .npz volumes live, for "
        "--fit-relaxometry.",
    )
    parser.add_argument(
        "--fit-what",
        choices=("both", "segmentations", "volumes"),
        default="both",
        help="Which datasets --fit-relaxometry should process (default: both).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Processes for --fit-relaxometry (default: cores - 2).",
    )
    parser.add_argument(
        "--mean-intensities",
        metavar="SEGMENTATIONS_DIR",
        help="Headless: read the already-saved masks in SEGMENTATIONS_DIR, "
        "average the voxels inside each one at every echo time, and write one "
        "Excel workbook with a sheet per sample.",
    )
    parser.add_argument(
        "--workbook",
        metavar="OUT.xlsx",
        help="Where --mean-intensities writes; defaults to "
        "segmentation_mean_intensities.xlsx inside SEGMENTATIONS_DIR.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code."""
    args = _build_parser().parse_args(argv)

    try:
        samples = get_available_samples(args.volumes)
    except DicomLoadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.list:
        for name in samples:
            echoes = get_available_echotimes(Path(args.volumes) / name)
            values = ", ".join(f"{e.value:g}" for e in echoes)
            source = "DICOM tag" if echoes and echoes[0].from_tag else "folder name"
            print(f"{name}: {len(echoes)} echo times (from {source})")
            print(f"  TE (ms): {values}")
        return 0

    if args.fit_relaxometry:
        from .relaxometry import run_segmentation_fits, run_volume_fits

        out_root = Path(args.fit_relaxometry)

        def announce(_done: int, _total: int, message: str) -> None:
            print(f"  fitting {message}", flush=True)

        reports = []
        if args.fit_what in ("both", "segmentations"):
            if not args.segmentations:
                print(
                    "error: --segmentations is required to fit segmentations",
                    file=sys.stderr,
                )
                return 1
            reports += run_segmentation_fits(
                args.segmentations,
                out_root / "Segmentations",
                samples=args.only,
                workers=args.workers,
                progress=announce,
            )
        if args.fit_what in ("both", "volumes"):
            reports += run_volume_fits(
                args.volumes,
                out_root / "Volumes",
                samples=args.only,
                workers=args.workers,
                progress=announce,
            )

        print()
        for report in reports:
            print(report.summary)
        print(f"\nwrote {len(reports)} datasets under {out_root}")
        return 0

    if args.mean_intensities:
        from .summary import export_mean_intensities

        def tick(done: int, total: int, message: str) -> None:
            print(f"  [{done}/{total}] {message}", flush=True)

        try:
            path, tables = export_mean_intensities(
                args.volumes,
                args.mean_intensities,
                workbook=args.workbook,
                samples=args.only,
                progress=tick,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

        print(f"\nwrote {path}")
        for sample, frame in tables.items():
            columns = [c for c in frame.columns if c != "EchoTime"]
            print(f"  sheet {sample}: {len(frame)} echo times x {len(columns)} segmentations")
        return 0

    if args.export_segmentations:
        from .export import export_all

        if args.only:
            unknown = [name for name in args.only if name not in samples]
            if unknown:
                print(
                    f"error: unknown sample(s) {', '.join(unknown)}. "
                    f"Available: {', '.join(samples)}",
                    file=sys.stderr,
                )
                return 1

        def show(done: int, total: int, message: str) -> None:
            print(f"  [{done:3d}/{total}] {message}", flush=True)

        reports = export_all(
            args.volumes,
            args.export_segmentations,
            samples=args.only,
            preset=None,
            write_sparse=not (args.dense_only or args.images_only),
            write_dense=not (args.sparse_only or args.images_only),
            write_images=not args.no_images,
            progress=show,
        )
        print()
        total_bytes = 0
        for report in reports:
            print(report.summary)
            total_bytes += report.bytes_written
            for echo_time, reason in report.skipped:
                print(f"  ! TE {echo_time:g} ms skipped: {reason}")
        print(f"\nwrote {total_bytes / 1e6:.1f} MB to {args.export_segmentations}")
        return 0

    if args.sample and args.sample not in samples:
        print(
            f"error: no sample named {args.sample!r}. Available: {', '.join(samples)}",
            file=sys.stderr,
        )
        return 1

    from .ui import run

    return run(args.volumes, sample=args.sample, cache_size=args.cache_size)


if __name__ == "__main__":
    raise SystemExit(main())
