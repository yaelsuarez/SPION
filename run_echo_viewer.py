#!/usr/bin/env python3
"""Standalone launcher for the echo-time DICOM viewer.

    python run_echo_viewer.py "/path/to/Volumes"
    python run_echo_viewer.py "/path/to/Volumes" --sample Syringes
    python run_echo_viewer.py "/path/to/Volumes" --list

Equivalent to ``python -m echoviewer``, but runnable without setting
PYTHONPATH first.
"""

import sys
from pathlib import Path

# Allow running from anywhere without installing this package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from echoviewer.__main__ import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
