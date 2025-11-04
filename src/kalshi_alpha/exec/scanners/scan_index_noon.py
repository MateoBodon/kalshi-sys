"""CLI scanner for noon index ladders (INXU, NASDAQ100U)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .index_scan_common import (
    ScannerConfig,
    build_parser,
    parse_timestamp,
    run_index_scan,
)

DEFAULT_SERIES: Sequence[str] = ("INXU", "NASDAQ100U")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser(DEFAULT_SERIES)
    args = parser.parse_args(argv)
    timestamp = parse_timestamp(args.now)
    config = ScannerConfig(
        series=tuple(s.upper() for s in args.series),
        min_ev=float(args.min_ev),
        max_bins=int(args.max_bins),
        contracts=int(args.contracts),
        kelly_cap=float(args.kelly_cap),
        offline=bool(args.offline),
        fixtures_root=Path(args.fixtures_root),
        output_root=Path(args.output_root),
        run_label="index_noon",
        timestamp=timestamp,
    )
    run_index_scan(config)


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    main()
