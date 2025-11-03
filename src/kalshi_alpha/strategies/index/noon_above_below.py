"""Backward-compatible shim for hourly index above/below strategy."""

from __future__ import annotations

from .hourly_above_below import HOURLY_CALIBRATION_PATH, HourlyInputs, pmf

NOON_CALIBRATION_PATH = HOURLY_CALIBRATION_PATH
NoonInputs = HourlyInputs

__all__ = ["NOON_CALIBRATION_PATH", "NoonInputs", "pmf"]
