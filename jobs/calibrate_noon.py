"""Backward-compatible shim for hourly calibration CLI."""

from __future__ import annotations

from . import calibrate_hourly as _calibrate_hourly

parse_args = _calibrate_hourly.parse_args
main = _calibrate_hourly.main
_write_params = _calibrate_hourly._write_params

__all__ = ["main", "parse_args", "_write_params"]


if __name__ == "__main__":  # pragma: no cover - legacy CLI entry
    main()
