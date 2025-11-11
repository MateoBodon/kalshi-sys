"""Shared datastore paths."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
PROC_ROOT = DATA_ROOT / "proc"
BOOTSTRAP_ROOT = DATA_ROOT / "bootstrap"
REPORTS_ROOT = PROJECT_ROOT / "reports"
CALIB_REPORTS_ROOT = REPORTS_ROOT / "calib"
INDEX_PMF_ROOT = PROC_ROOT / "calib" / "index_pmf"

for path in (RAW_ROOT, PROC_ROOT, BOOTSTRAP_ROOT, CALIB_REPORTS_ROOT, INDEX_PMF_ROOT):
    path.mkdir(parents=True, exist_ok=True)
