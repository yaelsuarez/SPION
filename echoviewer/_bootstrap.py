"""Locate the original ``dicomview`` package and put it on ``sys.path``.

This tool deliberately does not copy or modify the original viewer. It imports
it. Because the original lives outside this directory and is not pip-installed,
its parent folder has to be added to ``sys.path`` before the first import.

Search order:

1. ``$DICOMVIEW_HOME`` - set this to override everything else.
2. A few paths relative to this file, covering the normal repository layout.
3. Whatever is already importable (if you installed it yourself).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: Environment variable that overrides the search.
ENV_VAR = "DICOMVIEW_HOME"

_HERE = Path(__file__).resolve()


def _candidate_roots() -> list[Path]:
    """Directories that might contain the ``dicomview`` package."""
    candidates: list[Path] = []

    override = os.environ.get(ENV_VAR)
    if override:
        candidates.append(Path(override).expanduser().resolve())

    # .../uuutils/MRI/Project II/dicom_viewer/echoviewer/_bootstrap.py
    #     parents[1] = dicom_viewer (this tool)
    #     parents[4] = uuutils
    for up, tail in ((4, "dicom_viewer"), (3, "dicom_viewer"), (5, "dicom_viewer")):
        if len(_HERE.parents) > up:
            candidates.append(_HERE.parents[up] / tail)

    return candidates


def ensure_dicomview_importable() -> Path:
    """Make ``import dicomview`` work, and return the directory it was found in.

    Raises:
        ImportError: If the original viewer cannot be located, with the paths
            that were tried and how to point at it explicitly.
    """
    try:  # already importable - nothing to do
        import dicomview  # noqa: F401

        return Path(dicomview.__file__).resolve().parent.parent
    except ImportError:
        pass

    tried: list[Path] = []
    for root in _candidate_roots():
        tried.append(root)
        if (root / "dicomview" / "__init__.py").is_file():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            return root

    listing = "\n  ".join(str(p) for p in tried) or "  (none)"
    raise ImportError(
        "Could not find the original 'dicomview' package. Looked in:\n  "
        f"{listing}\n"
        f"Set {ENV_VAR} to the directory that contains the 'dicomview' folder, "
        "e.g. export "
        f"{ENV_VAR}=<path to the dicom_viewer checkout>"
    )


#: Resolved on import so that every module in this package can simply
#: ``from ._bootstrap import DICOMVIEW_ROOT`` and then import dicomview.
DICOMVIEW_ROOT = ensure_dicomview_importable()
