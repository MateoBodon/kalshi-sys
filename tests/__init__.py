"""Test package for Kalshi Alpha with src/ path bootstrapping."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
SRC_STR = str(SRC_ROOT)
if SRC_ROOT.exists() and SRC_STR not in sys.path:
    sys.path.insert(0, SRC_STR)
