"""Backward-compatible shim for hourly index ladder scanning."""

from __future__ import annotations

from .scan_index_hourly import (
    HOURLY_CALIBRATION_PATH,
    IndexScanResult,
    QuoteOpportunity,
    evaluate_hourly,
)

NOON_CALIBRATION_PATH = HOURLY_CALIBRATION_PATH
evaluate_noon = evaluate_hourly

__all__ = ["NOON_CALIBRATION_PATH", "QuoteOpportunity", "IndexScanResult", "evaluate_noon"]
