"""Fast offline index scan helpers used by --fast-fixtures."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime, time
from pathlib import Path
from typing import Iterable, Sequence

import polars as pl

DEFAULT_FAST_ROOT = Path("tests/data_fixtures")
DEFAULT_OUTPUT_ROOT = Path("reports/index_ladders")
DEFAULT_HOURLY_SERIES: tuple[str, ...] = ("INXU", "NASDAQ100U")
DEFAULT_CLOSE_SERIES: tuple[str, ...] = ("INX", "NASDAQ100")
DEFAULT_TARGET_HOURS = (10, 11, 12, 13, 14, 15, 16)

SERIES_TO_SYMBOL: dict[str, str] = {
    "INX": "I_SPX",
    "INXU": "I_SPX",
    "NASDAQ100": "I_NDX",
    "NASDAQ100U": "I_NDX",
}

CSV_HEADER = [
    "series",
    "market",
    "strike",
    "side",
    "event",
    "q_yes",
    "model_probability",
    "market_probability",
    "ev_after_fees",
    "ev_per_contract",
    "contracts",
    "alpha",
    "slippage",
    "delta_bps",
]


@dataclass(frozen=True)
class FastIndexConfig:
    series: tuple[str, ...]
    target_hours: tuple[int, ...]
    fixtures_root: Path
    output_root: Path
    base_timestamp: datetime
    contracts: int = 1
    min_ev: float = 0.05


def _parse_timestamp(value: str | None) -> datetime:
    if not value:
        return datetime.now(tz=UTC)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _load_minutes(symbol: str, fixtures_root: Path) -> pl.DataFrame:
    path = fixtures_root / "index_fast" / f"{symbol}_minutes.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Fast Polygon fixture missing: {path}")
    return pl.read_parquet(path)


def _latest_price(frame: pl.DataFrame) -> tuple[datetime, float]:
    if frame.is_empty():
        raise ValueError("Fast fixtures contain no rows")
    last_row = frame.sort("timestamp").tail(1)
    ts = last_row.item(0, "timestamp")
    price = float(last_row.item(0, "close"))
    return ts, price


def _build_row(
    *,
    series: str,
    target_hour: int,
    timestamp: datetime,
    price: float,
    contracts: int,
    min_ev: float,
) -> dict[str, object]:
    strike = round(price / 25.0) * 25.0
    yes_price = 0.45
    model_prob = 0.5
    market_prob = yes_price
    maker_ev = max(min_ev, (model_prob - market_prob))
    delta_component = maker_ev * 0.3
    return {
        "series": series,
        "market": f"{series.upper()}_{target_hour:02d}00_FAST",
        "strike": float(strike),
        "side": "YES",
        "event": f"{series.upper()}_{target_hour:02d}00",
        "q_yes": float(yes_price),
        "model_probability": float(model_prob),
        "market_probability": float(market_prob),
        "ev_after_fees": float(maker_ev * contracts),
        "ev_per_contract": float(maker_ev),
        "contracts": int(contracts),
        "alpha": 0.3,
        "slippage": 0.0,
        "delta_bps": float(delta_component * 10000.0),
        "timestamp": timestamp,
    }


def _write_csv(rows: Sequence[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(CSV_HEADER) + "\n")
        for row in rows:
            handle.write(
                ",".join(
                    [
                        str(row.get("series", "")),
                        str(row.get("market", "")),
                        f"{float(row.get('strike', 0.0)):.2f}",
                        str(row.get("side", "")),
                        str(row.get("event", "")),
                        f"{float(row.get('q_yes', 0.0)):.4f}",
                        f"{float(row.get('model_probability', 0.0)):.6f}",
                        f"{float(row.get('market_probability', 0.0)):.6f}",
                        f"{float(row.get('ev_after_fees', 0.0)):.6f}",
                        f"{float(row.get('ev_per_contract', 0.0)):.6f}",
                        str(int(row.get("contracts", 0))),
                        f"{float(row.get('alpha', 0.0)):.6f}",
                        f"{float(row.get('slippage', 0.0)):.6f}",
                        f"{float(row.get('delta_bps', 0.0)):.4f}",
                    ]
                )
                + "\n"
            )


def _resolve_series(raw: Iterable[str] | None, *, default: tuple[str, ...]) -> tuple[str, ...]:
    if raw is None:
        return default
    cleaned = tuple(entry.strip().upper() for entry in raw if entry and entry.strip())
    return cleaned or default


def run_fast_hourly(args: argparse.Namespace) -> None:
    config = FastIndexConfig(
        series=_resolve_series(getattr(args, "series", None), default=DEFAULT_HOURLY_SERIES),
        target_hours=tuple(sorted({hour % 24 for hour in (args.target_hours or DEFAULT_TARGET_HOURS)})),
        fixtures_root=Path(getattr(args, "fixtures_root", DEFAULT_FAST_ROOT)),
        output_root=Path(getattr(args, "output_root", DEFAULT_OUTPUT_ROOT)),
        base_timestamp=_parse_timestamp(getattr(args, "now", None)),
        contracts=max(int(getattr(args, "contracts", 1)), 1),
        min_ev=float(getattr(args, "min_ev", 0.05)),
    )
    for hour in config.target_hours:
        timestamp_label = config.base_timestamp.strftime("%Y%m%dT%H%M%SZ")
        hour_dir = config.output_root / f"{hour:02d}00"
        for series in config.series:
            symbol = SERIES_TO_SYMBOL.get(series.upper())
            if symbol is None:
                continue
            frame = _load_minutes(symbol, config.fixtures_root)
            ts, price = _latest_price(frame)
            row = _build_row(
                series=series.upper(),
                target_hour=hour,
                timestamp=ts,
                price=price,
                contracts=config.contracts,
                min_ev=config.min_ev,
            )
            series_dir = hour_dir / series.upper()
            output_path = series_dir / f"{timestamp_label}.csv"
            _write_csv([row], output_path)


def run_fast_close(args: argparse.Namespace) -> None:
    config = FastIndexConfig(
        series=_resolve_series(getattr(args, "series", None), default=DEFAULT_CLOSE_SERIES),
        target_hours=(16,),
        fixtures_root=Path(getattr(args, "fixtures_root", DEFAULT_FAST_ROOT)),
        output_root=Path(getattr(args, "output_root", DEFAULT_OUTPUT_ROOT)),
        base_timestamp=_parse_timestamp(getattr(args, "now", None)),
        contracts=max(int(getattr(args, "contracts", 1)), 1),
        min_ev=float(getattr(args, "min_ev", 0.05)),
    )
    timestamp_label = config.base_timestamp.strftime("%Y%m%dT%H%M%SZ")
    for series in config.series:
        symbol = SERIES_TO_SYMBOL.get(series.upper())
        if symbol is None:
            continue
        frame = _load_minutes(symbol, config.fixtures_root)
        ts, price = _latest_price(frame)
        row = _build_row(
            series=series.upper(),
            target_hour=16,
            timestamp=ts,
            price=price,
            contracts=config.contracts,
            min_ev=config.min_ev,
        )
        series_dir = config.output_root / "1600" / series.upper()
        _write_csv([row], series_dir / f"{timestamp_label}.csv")


__all__ = ["run_fast_hourly", "run_fast_close"]
