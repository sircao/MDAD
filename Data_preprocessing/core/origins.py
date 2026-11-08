"""Path adapter for original project source roots.

Pipeline tools (pipelines/) reuse modules from three original projects. This module
provides each project's source root and an ``ensure`` helper that adds the root to
``sys.path``, so tools can ``import`` original project modules directly inside
``run()``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# E:\\DATA_PREPROCESSING
DATA_ROOT = Path(__file__).resolve().parent.parent.parent

# RS-TOUTER (TS-Router-D8FF)
RS_ROUTER_ROOT = DATA_ROOT / "TS-Router-D8FF"

# Multi-AXIS (Multi-AXIS-B1DC/src, package name multi_axis)
MULTI_AXIS_SRC = DATA_ROOT / "Multi-AXIS-B1DC" / "src"

# MTS dataset synthesis (Multi-AXIS-B1DC/MTS_Data/src, package name mvaxis)
MTS_SRC = DATA_ROOT / "Multi-AXIS-B1DC" / "MTS_Data" / "src"


def ensure(path: Path) -> Path:
    """Add a source root to sys.path (if not already present) and return it."""
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
    return path
