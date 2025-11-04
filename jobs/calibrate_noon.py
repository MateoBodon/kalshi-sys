"""Deprecated shim for hourly calibration; retains CLI compatibility."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

from .calibrate_hourly import main as _hourly_main
from .calibrate_hourly import parse_args as _parse_args


def parse_args(argv: Sequence[str] | None = None):  # type: ignore[override]
    warnings.warn(
        "calibrate_noon is deprecated; use calibrate_hourly",
        DeprecationWarning,
        stacklevel=2,
    )
    return _parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - thin shim
    warnings.warn(
        "calibrate_noon redirects to calibrate_hourly and writes hourly params",
        DeprecationWarning,
        stacklevel=2,
    )
    _hourly_main(argv)


__all__ = ["main", "parse_args"]


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
