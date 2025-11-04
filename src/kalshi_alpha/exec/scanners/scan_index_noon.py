"""Deprecated alias for hourly index ladder scanner CLI."""

from __future__ import annotations

import warnings
from collections.abc import Sequence

from .scan_index_hourly import DEFAULT_SERIES
from .scan_index_hourly import main as _hourly_main


def main(argv: Sequence[str] | None = None) -> None:
    warnings.warn(
        "scan_index_noon is deprecated; use scan_index_hourly",
        DeprecationWarning,
        stacklevel=2,
    )
    _hourly_main(argv)


__all__ = ["DEFAULT_SERIES", "main"]
