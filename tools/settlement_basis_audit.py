"""Settlement basis audit for index ladder windows (Polygon vs Kalshi expiration value)."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import polars as pl

from kalshi_alpha.core.kalshi_api import KalshiPublicClient
from kalshi_alpha.datastore.paths import PROC_ROOT, REPORTS_ROOT
from kalshi_alpha.drivers.polygon_index.client import MinuteBar, PolygonIndicesClient
from kalshi_alpha.drivers.polygon_index.symbols import resolve_series
from kalshi_alpha.markets.discovery import DiscoveredMarket, WindowDiscovery, discover_markets_for_day
from kalshi_alpha.sched.windows import TradingWindow, windows_for_day

ET = ZoneInfo("America/New_York")
SUPPORTED_SERIES = {"INX", "INXU", "NASDAQ100", "NASDAQ100U"}
DEFAULT_FIXTURES_ROOT = Path("tests/fixtures/settlement_basis")
KALSHI_VALUE_FIELDS = (
    "settlement_value",
    "settlement_price",
    "expiration_value",
    "expiration_price",
    "final_value",
    "final_price",
)


@dataclass(frozen=True)
class PolygonWindowValue:
    value: float | None
    timestamp_utc: datetime | None
    source: str


@dataclass(frozen=True)
class KalshiSettlementValue:
    value: float
    source_field: str


def _parse_day(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--day must be YYYY-MM-DD") from exc


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Polygon index prints to Kalshi expiration values.")
    parser.add_argument("--day", type=_parse_day, required=True, help="Trading day (YYYY-MM-DD).")
    parser.add_argument(
        "--series",
        type=str,
        required=True,
        choices=sorted(SUPPORTED_SERIES),
        help="Index ladder series (INX, INXU, NASDAQ100, NASDAQ100U).",
    )
    parser.add_argument(
        "--out-report",
        type=str,
        default=None,
        help="Path for markdown report (default: reports/settlement_basis/<day>_<series>.md).",
    )
    parser.add_argument(
        "--out-data",
        type=str,
        default=None,
        help="Path for dataset output (default: data/proc/settlement_basis/<day>_<series>.parquet).",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="If dataset exists, regenerate report from it without API calls.",
    )
    parser.add_argument(
        "--offline-fixtures",
        nargs="?",
        const=str(DEFAULT_FIXTURES_ROOT),
        default=None,
        help="Use offline fixture JSON (default: tests/fixtures/settlement_basis).",
    )
    return parser.parse_args(argv)


def _resolve_paths(day: date, series: str, *, out_report: str | None, out_data: str | None) -> tuple[Path, Path]:
    report_path = Path(out_report) if out_report else REPORTS_ROOT / "settlement_basis" / f"{day.isoformat()}_{series.upper()}.md"
    data_path = Path(out_data) if out_data else PROC_ROOT / "settlement_basis" / f"{day.isoformat()}_{series.upper()}.parquet"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    return report_path, data_path


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    normalized = raw.replace("Z", "+00:00") if raw.endswith("Z") else raw
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _load_polygon_offline_values(fixtures_root: Path, *, day: date, series: str) -> dict[str, PolygonWindowValue]:
    path = fixtures_root / "polygon" / f"polygon_{series.upper()}_{day.isoformat()}.json"
    if not path.exists():
        raise FileNotFoundError(f"Polygon fixture missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    values: dict[str, PolygonWindowValue] = {}
    for entry in payload.get("values", []):
        label = str(entry.get("window_label") or entry.get("label") or "").strip()
        if not label:
            continue
        value = _safe_float(entry.get("value") or entry.get("price") or entry.get("close"))
        ts = _parse_iso(entry.get("timestamp_utc") or entry.get("ts_utc"))
        source = str(entry.get("source") or "offline_fixture")
        values[label] = PolygonWindowValue(value=value, timestamp_utc=ts, source=source)
    return values


def _polygon_values_online(series: str, windows: list[TradingWindow]) -> dict[str, PolygonWindowValue]:
    if not windows:
        return {}
    symbol = resolve_series(series).polygon_ticker
    start_utc = min(window.start_utc for window in windows) - timedelta(minutes=1)
    end_utc = max(window.target_utc for window in windows) + timedelta(minutes=1)
    client = PolygonIndicesClient()
    bars = client.fetch_minute_bars(symbol, start_utc, end_utc)
    bars_sorted = sorted(bars, key=lambda bar: bar.timestamp)
    results: dict[str, PolygonWindowValue] = {}
    idx = 0
    last_bar: MinuteBar | None = None
    for window in sorted(windows, key=lambda entry: entry.target_utc):
        while idx < len(bars_sorted) and bars_sorted[idx].timestamp <= window.target_utc:
            last_bar = bars_sorted[idx]
            idx += 1
        if last_bar is None:
            results[window.label] = PolygonWindowValue(value=None, timestamp_utc=None, source="polygon_minute_close")
        else:
            results[window.label] = PolygonWindowValue(
                value=float(last_bar.close),
                timestamp_utc=last_bar.timestamp,
                source="polygon_minute_close",
            )
    return results


def _find_field(payload: Any, key: str, path: str = "") -> tuple[float, str] | None:
    if isinstance(payload, dict):
        for child_key, child_value in payload.items():
            if child_key == key:
                value = _safe_float(child_value)
                if value is not None:
                    return value, f"{path}{child_key}"
        for child_key, child_value in payload.items():
            if isinstance(child_value, (dict, list)):
                result = _find_field(child_value, key, f"{path}{child_key}.")
                if result is not None:
                    return result
    elif isinstance(payload, list):
        for idx, entry in enumerate(payload):
            result = _find_field(entry, key, f"{path}[{idx}].")
            if result is not None:
                return result
    return None


def _extract_kalshi_value(payload: dict[str, Any]) -> KalshiSettlementValue | None:
    for key in KALSHI_VALUE_FIELDS:
        found = _find_field(payload, key)
        if found is not None:
            value, path = found
            return KalshiSettlementValue(value=value, source_field=path)
    return None


def _collect_strikes(client: KalshiPublicClient, event_id: str) -> list[float]:
    markets = client.get_markets(event_id)
    strikes: set[float] = set()
    for market in markets:
        strikes.update(float(value) for value in market.ladder_strikes)
    return sorted(strikes)


def _nearest_strike(value: float | None, strikes: Iterable[float]) -> tuple[float | None, float | None]:
    if value is None:
        return None, None
    strike_list = list(strikes)
    if not strike_list:
        return None, None
    nearest = min(strike_list, key=lambda strike: abs(value - strike))
    margin = abs(value - nearest)
    return float(nearest), float(margin)


def _load_frame(path: Path) -> pl.DataFrame:
    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    if path.suffix in {".jsonl", ".ndjson"}:
        return pl.read_ndjson(path)
    if path.suffix == ".json":
        return pl.read_json(path)
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def _write_frame(frame: pl.DataFrame, path: Path) -> None:
    if path.suffix == ".parquet":
        frame.write_parquet(path)
        return
    if path.suffix in {".jsonl", ".ndjson"}:
        frame.write_ndjson(path)
        return
    if path.suffix == ".json":
        frame.write_json(path)
        return
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _format_float(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.4f}"


def _series_stats(series: pl.Series) -> dict[str, float | None]:
    if series.is_empty():
        return {"mean": None, "median": None, "p95": None, "p99": None}
    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "p95": float(series.quantile(0.95, "nearest")),
        "p99": float(series.quantile(0.99, "nearest")),
    }


def _render_report(
    frame: pl.DataFrame,
    *,
    day: date,
    series: str,
    dataset_path: Path,
    command: str,
) -> str:
    total_rows = frame.height
    basis_series = frame.get_column("basis").drop_nulls() if "basis" in frame.columns else pl.Series([])
    abs_series = basis_series.abs() if not basis_series.is_empty() else pl.Series([])
    basis_stats = _series_stats(basis_series)
    abs_stats = _series_stats(abs_series)
    flip_count = frame.filter(pl.col("flip_risk") == True).height if "flip_risk" in frame.columns else 0
    missing_kalshi = frame.filter(pl.col("kalshi_value").is_null()).height if "kalshi_value" in frame.columns else total_rows
    missing_polygon = frame.filter(pl.col("polygon_value").is_null()).height if "polygon_value" in frame.columns else total_rows

    lines: list[str] = []
    lines.append(f"# Settlement Basis Audit — {series.upper()} {day.isoformat()}")
    lines.append("")
    lines.append(f"- Generated: {datetime.now(tz=UTC).isoformat()}")
    lines.append(f"- Dataset: {dataset_path}")
    lines.append(f"- Command: `{command}`")
    lines.append(f"- Git SHA: `{_git_sha()}`")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Windows: {total_rows}")
    lines.append(f"- Missing Kalshi values: {missing_kalshi}")
    lines.append(f"- Missing Polygon values: {missing_polygon}")
    lines.append(f"- Flip-risk windows: {flip_count}")
    lines.append(
        "- Basis stats (value=Polygon-Kalshi): "
        f"mean={_format_float(basis_stats['mean'])}, "
        f"median={_format_float(basis_stats['median'])}, "
        f"p95={_format_float(basis_stats['p95'])}, "
        f"p99={_format_float(basis_stats['p99'])}"
    )
    lines.append(
        "- Abs(basis) stats: "
        f"mean={_format_float(abs_stats['mean'])}, "
        f"median={_format_float(abs_stats['median'])}, "
        f"p95={_format_float(abs_stats['p95'])}, "
        f"p99={_format_float(abs_stats['p99'])}"
    )
    lines.append("")
    lines.append("Polygon is not settlement truth; Kalshi expiration value is the reference for settlement.")
    lines.append("")

    lines.append("## Top Windows by |Basis|")
    if basis_series.is_empty():
        lines.append("_No basis values available._")
    else:
        top = (
            frame.filter(pl.col("basis").is_not_null())
            .with_columns(pl.col("basis").abs().alias("abs_basis"))
            .sort("abs_basis", descending=True)
            .head(5)
        )
        headers = ["window", "window_ts_et", "polygon_value", "kalshi_value", "basis", "abs_basis"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in top.iter_rows(named=True):
            lines.append(
                "| {window} | {window_ts_et} | {polygon_value} | {kalshi_value} | {basis} | {abs_basis} |".format(
                    window=row.get("window_label", ""),
                    window_ts_et=row.get("window_ts_et", ""),
                    polygon_value=_format_float(row.get("polygon_value")),
                    kalshi_value=_format_float(row.get("kalshi_value")),
                    basis=_format_float(row.get("basis")),
                    abs_basis=_format_float(row.get("abs_basis")),
                )
            )
    lines.append("")

    lines.append("## Flip-Risk Windows")
    flips = frame.filter(pl.col("flip_risk") == True) if "flip_risk" in frame.columns else pl.DataFrame()
    if flips.is_empty():
        lines.append("_No flip-risk windows flagged._")
    else:
        headers = ["window", "window_ts_et", "polygon_value", "kalshi_value", "basis", "nearest_strike", "margin"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in flips.iter_rows(named=True):
            lines.append(
                "| {window} | {window_ts_et} | {polygon_value} | {kalshi_value} | {basis} | {nearest_strike} | {margin} |".format(
                    window=row.get("window_label", ""),
                    window_ts_et=row.get("window_ts_et", ""),
                    polygon_value=_format_float(row.get("polygon_value")),
                    kalshi_value=_format_float(row.get("kalshi_value")),
                    basis=_format_float(row.get("basis")),
                    nearest_strike=_format_float(row.get("nearest_strike")),
                    margin=_format_float(row.get("nearest_strike_margin")),
                )
            )
    lines.append("")
    return "\n".join(lines)


def _windows_for_series(day: date, series: str) -> list[TradingWindow]:
    windows = windows_for_day(day)
    filtered = [window for window in windows if series.upper() in window.series]
    return filtered


def _discover_markets(
    client: KalshiPublicClient,
    *,
    day: date,
    series: str,
) -> tuple[dict[str, DiscoveredMarket], dict[str, WindowDiscovery]]:
    discoveries = discover_markets_for_day(client, trading_day=day, series=[series], status=None)
    market_by_label: dict[str, DiscoveredMarket] = {}
    discovery_by_label: dict[str, WindowDiscovery] = {}
    for discovery in discoveries:
        discovery_by_label[discovery.label] = discovery
        for market in discovery.markets:
            if market.series.upper() == series.upper():
                market_by_label[discovery.label] = market
    return market_by_label, discovery_by_label


def _build_dataset(
    *,
    day: date,
    series: str,
    offline_fixtures: Path | None,
) -> pl.DataFrame:
    windows = _windows_for_series(day, series)
    if not windows:
        raise RuntimeError(f"No windows available for {series} on {day.isoformat()}")

    if offline_fixtures:
        kalshi_client = KalshiPublicClient(offline_dir=offline_fixtures / "kalshi", use_offline=True)
        polygon_values = _load_polygon_offline_values(offline_fixtures, day=day, series=series)
    else:
        kalshi_client = KalshiPublicClient()
        polygon_values = _polygon_values_online(series, windows)

    market_by_label, _ = _discover_markets(kalshi_client, day=day, series=series)
    kalshi_cache: dict[str, KalshiSettlementValue] = {}
    strikes_cache: dict[str, list[float]] = {}

    rows: list[dict[str, object]] = []
    for window in windows:
        market = market_by_label.get(window.label)
        event_id = market.event_id if market else None
        kalshi_value: float | None = None
        kalshi_source: str | None = None
        strikes: list[float] = []
        if event_id:
            if event_id not in kalshi_cache:
                payload = kalshi_client.get_event_detail(event_id)
                settlement = _extract_kalshi_value(payload)
                if settlement is None:
                    raise RuntimeError(
                        f"Kalshi expiration value missing for event {event_id}. "
                        "Confirm the settlement field in the /events/{event_id} payload."
                    )
                kalshi_cache[event_id] = settlement
            if event_id not in strikes_cache:
                strikes_cache[event_id] = _collect_strikes(kalshi_client, event_id)
            kalshi_value = kalshi_cache[event_id].value
            kalshi_source = kalshi_cache[event_id].source_field
            strikes = strikes_cache.get(event_id, [])

        polygon = polygon_values.get(window.label, PolygonWindowValue(value=None, timestamp_utc=None, source=""))
        basis = None
        if kalshi_value is not None and polygon.value is not None:
            basis = float(polygon.value) - float(kalshi_value)

        nearest_strike, margin = _nearest_strike(polygon.value, strikes)
        flip_risk = False
        if basis is not None and margin is not None:
            flip_risk = abs(basis) >= margin

        rows.append(
            {
                "day": day,
                "series": series.upper(),
                "window_label": window.label,
                "window_ts_et": window.target_et.isoformat(),
                "window_ts_utc": window.target_utc.isoformat(),
                "kalshi_value": kalshi_value,
                "kalshi_source_field": kalshi_source,
                "kalshi_market_or_event_id": event_id,
                "polygon_value": polygon.value,
                "polygon_source": polygon.source,
                "polygon_ts_utc": polygon.timestamp_utc.isoformat() if polygon.timestamp_utc else None,
                "basis": basis,
                "nearest_strike": nearest_strike,
                "nearest_strike_margin": margin,
                "flip_risk": flip_risk,
            }
        )

    frame = pl.DataFrame(rows)
    return frame


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    report_path, data_path = _resolve_paths(args.day, args.series, out_report=args.out_report, out_data=args.out_data)
    raw_args = argv if argv is not None else sys.argv[1:]
    command = " ".join(shlex.quote(part) for part in ["python", "tools/settlement_basis_audit.py", *[str(a) for a in raw_args]])

    if args.use_cache:
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found for --use-cache: {data_path}")
        frame = _load_frame(data_path)
        report = _render_report(frame, day=args.day, series=args.series, dataset_path=data_path, command=command)
        report_path.write_text(report, encoding="utf-8")
        print(f"[settlement_basis] report written to {report_path}")
        return

    offline_root = Path(args.offline_fixtures).resolve() if args.offline_fixtures else None
    frame = _build_dataset(day=args.day, series=args.series, offline_fixtures=offline_root)
    _write_frame(frame, data_path)
    report = _render_report(frame, day=args.day, series=args.series, dataset_path=data_path, command=command)
    report_path.write_text(report, encoding="utf-8")
    print(f"[settlement_basis] dataset written to {data_path}")
    print(f"[settlement_basis] report written to {report_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
