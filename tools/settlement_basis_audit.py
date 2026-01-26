"""Settlement basis audit for index ladder windows (Polygon vs Kalshi expiration value)."""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import polars as pl

from kalshi_alpha.brokers.kalshi.http_client import KalshiHttpClient, KalshiHttpError
from kalshi_alpha.core.kalshi_api import KalshiPublicClient, Market
from kalshi_alpha.datastore.paths import PROC_ROOT, REPORTS_ROOT
from kalshi_alpha.drivers.polygon_index.client import MinuteBar, PolygonIndicesClient
from kalshi_alpha.drivers.polygon_index.symbols import resolve_series
from kalshi_alpha.markets.discovery import DiscoveredMarket, WindowDiscovery, discover_markets_for_day
from kalshi_alpha.sched.windows import TradingWindow, windows_for_day
from kalshi_alpha.utils.env import load_env

ET = ZoneInfo("America/New_York")
SUPPORTED_SERIES = {"INX", "INXU", "NASDAQ100", "NASDAQ100U"}
SUPPORTED_SERIES_ORDER = ("INX", "INXU", "NASDAQ100", "NASDAQ100U")
DEFAULT_FIXTURES_ROOT = Path("tests/fixtures/settlement_basis")
DEFAULT_QUOTE_DISTANCE = 0.5
EXEC_DEFAULTS_PATH = PROJECT_ROOT / "data" / "reference" / "index_execution_defaults.json"
KALSHI_SERIES_TICKERS = {
    "INX": "KXINX",
    "INXU": "KXINXU",
    "NASDAQ100": "KXNASDAQ100",
    "NASDAQ100U": "KXNASDAQ100U",
}
EVENT_TICKER_PATTERN = re.compile(
    r"-(?P<year>\d{2})(?P<month>[A-Z]{3})(?P<day>\d{2})H(?P<hour>\d{2})(?P<minute>\d{2})"
)
EVENT_TICKER_MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}
EVENT_MATCH_TOLERANCE_SECONDS = 90
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


def _kalshi_series_ticker(series: str | None) -> str | None:
    if series is None:
        return None
    normalized = series.upper().strip()
    if normalized.startswith("KX"):
        return normalized
    return KALSHI_SERIES_TICKERS.get(normalized, normalized)


def _close_time_from_ticker(ticker: str) -> datetime | None:
    match = EVENT_TICKER_PATTERN.search(ticker or "")
    if not match:
        return None
    month = EVENT_TICKER_MONTHS.get(match.group("month"))
    if month is None:
        return None
    year = 2000 + int(match.group("year"))
    day = int(match.group("day"))
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    return datetime(year, month, day, hour, minute, tzinfo=ET).astimezone(UTC)


class _KalshiAuthenticatedClient:
    """Adapter for Kalshi trade-api/v2 using the KalshiHttpClient interface."""

    def __init__(self, *, base_url: str | None = None) -> None:
        load_env()
        self._http = KalshiHttpClient(base_url=base_url or "https://api.elections.kalshi.com/trade-api/v2")

    def get_event_detail(self, event_id: str, *, force_refresh: bool = False) -> dict[str, Any]:
        try:
            response = self._http.get(f"/events/{event_id}")
        except KalshiHttpError as exc:
            raise RuntimeError(f"Kalshi event fetch failed for {event_id}: {exc}") from exc
        payload = response.json()
        if isinstance(payload, dict):
            return payload
        return {"event": payload}

    def get_markets(self, event_id: str, *, force_refresh: bool = False) -> list[Market]:
        raw_markets = self._fetch_markets(event_ticker=event_id, series_ticker=None, status=None, limit=None)
        return [Market.from_payload(_normalize_market_payload(item, series_ticker=None)) for item in raw_markets]

    def search_markets(  # noqa: PLR0913
        self,
        *,
        series_ticker: str | None = None,
        status: str | None = None,
        event_ticker: str | None = None,
        limit: int | None = None,
        force_refresh: bool = False,
    ) -> list[Market]:
        raw_markets = self._fetch_markets(
            series_ticker=series_ticker,
            status=status,
            event_ticker=event_ticker,
            limit=limit,
        )
        return [
            Market.from_payload(_normalize_market_payload(item, series_ticker=series_ticker))
            for item in raw_markets
        ]

    def _fetch_markets(
        self,
        *,
        series_ticker: str | None,
        status: str | None,
        event_ticker: str | None,
        limit: int | None,
    ) -> list[dict[str, Any]]:
        base_params: dict[str, Any] = {}
        if series_ticker:
            base_params["series_ticker"] = _kalshi_series_ticker(series_ticker)
        if event_ticker:
            base_params["event_ticker"] = event_ticker
        if limit is not None:
            base_params["limit"] = int(limit)

        status_candidates = [status] if status is not None else [None, "closed", "settled", "open"]
        if event_ticker and status is None:
            status_candidates = [None]
        results: list[dict[str, Any]] = []
        seen: set[str] = set()

        for status_value in status_candidates:
            params = dict(base_params)
            if status_value is not None:
                params["status"] = status_value
            try:
                markets = self._fetch_pages(params, paginate=limit is None)
            except KalshiHttpError:
                if status_value is not None:
                    continue
                raise
            for payload in markets:
                ticker = str(payload.get("ticker") or "")
                if ticker and ticker in seen:
                    continue
                if ticker:
                    seen.add(ticker)
                results.append(payload)

        return results

    def _fetch_pages(self, params: dict[str, Any], *, paginate: bool) -> list[dict[str, Any]]:
        cursor: str | None = None
        results: list[dict[str, Any]] = []
        while True:
            page_params = dict(params)
            if cursor:
                page_params["cursor"] = cursor
            response = self._http.get("/markets", params=page_params)
            payload = response.json()
            if not isinstance(payload, dict):
                break
            results.extend(payload.get("markets") or [])
            cursor = payload.get("cursor")
            if not paginate or not cursor:
                break
        return results

    def list_events(self, *, series_ticker: str, status: str | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"series_ticker": _kalshi_series_ticker(series_ticker)}
        if status:
            params["status"] = status
        cursor: str | None = None
        results: list[dict[str, Any]] = []
        while True:
            page_params = dict(params)
            if cursor:
                page_params["cursor"] = cursor
            response = self._http.get("/events", params=page_params)
            payload = response.json()
            if not isinstance(payload, dict):
                break
            results.extend(payload.get("events") or [])
            cursor = payload.get("cursor")
            if not cursor:
                break
        return results


def _normalize_market_payload(payload: dict[str, Any], *, series_ticker: str | None) -> dict[str, Any]:
    normalized = dict(payload)
    event_ticker = normalized.get("event_ticker")
    if event_ticker and "event_id" not in normalized:
        normalized["event_id"] = event_ticker
    if series_ticker and not normalized.get("series_ticker"):
        normalized["series_ticker"] = series_ticker.upper()
    if "close_time" in normalized and "close_ts" not in normalized:
        normalized["close_ts"] = normalized["close_time"]
    if "ladder_strikes" not in normalized:
        strike = _safe_float(normalized.get("floor_strike"))
        normalized["ladder_strikes"] = [strike] if strike is not None else []
    if "ladder_yes_prices" not in normalized:
        normalized["ladder_yes_prices"] = []
    return normalized


def _parse_day(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--day must be YYYY-MM-DD") from exc


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Polygon index prints to Kalshi expiration values.")
    parser.add_argument("--day", type=_parse_day, required=True, help="Trading day (YYYY-MM-DD).")
    series_group = parser.add_mutually_exclusive_group(required=True)
    series_group.add_argument(
        "--series",
        type=str,
        choices=sorted(SUPPORTED_SERIES),
        help="Index ladder series (INX, INXU, NASDAQ100, NASDAQ100U).",
    )
    series_group.add_argument(
        "--all-series",
        action="store_true",
        help="Run the audit for all supported series (INX, INXU, NASDAQ100, NASDAQ100U).",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Path for JSON summary (default: data/proc/basis/<SERIES>/<YYYY-MM-DD>.json).",
    )
    parser.add_argument(
        "--out-report",
        type=str,
        default=None,
        help="Path for markdown report (default: reports/basis/<SERIES>/<YYYY-MM-DD>.md).",
    )
    parser.add_argument(
        "--out-data",
        type=str,
        default=None,
        help="Optional path for dataset output (default: data/proc/settlement_basis/<day>_<series>.parquet).",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="If dataset exists, regenerate report/summary from it without API calls.",
    )
    parser.add_argument(
        "--offline-fixtures",
        nargs="?",
        const=str(DEFAULT_FIXTURES_ROOT),
        default=None,
        help="Use offline fixture JSON (default: tests/fixtures/settlement_basis).",
    )
    parser.add_argument("--archive-dir", type=Path, default=None, help="Archive outputs into this directory.")
    parser.add_argument("--runlog", type=Path, default=None, help="Alias for --archive-dir (run log folder).")
    return parser.parse_args(argv)


def _resolve_paths(
    day: date,
    series: str,
    *,
    out_report: str | None,
    out_data: str | None,
    out_json: str | None,
) -> tuple[Path, Path, Path]:
    series_slug = series.upper()
    report_path = (
        Path(out_report)
        if out_report
        else REPORTS_ROOT / "basis" / series_slug / f"{day.isoformat()}.md"
    )
    data_path = (
        Path(out_data)
        if out_data
        else PROC_ROOT / "settlement_basis" / f"{day.isoformat()}_{series_slug}.parquet"
    )
    json_path = (
        Path(out_json)
        if out_json
        else PROC_ROOT / "basis" / series_slug / f"{day.isoformat()}.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    return report_path, data_path, json_path


def _resolve_archive_dir(archive_dir: Path | None, runlog_dir: Path | None) -> Path | None:
    if archive_dir is None:
        return runlog_dir
    if runlog_dir is None:
        return archive_dir
    if archive_dir == runlog_dir:
        return archive_dir
    raise ValueError("Pass only one of --archive-dir or --runlog (or ensure they match).")


def _archive_outputs(archive_dir: Path, paths: Sequence[Path]) -> None:
    archive_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Artifact missing for archive: {path}")
        try:
            relative = path.resolve().relative_to(PROJECT_ROOT)
            target = archive_dir / relative
        except ValueError:
            target = archive_dir / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


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


def _median_spacing(strikes: Iterable[float]) -> float | None:
    values = sorted(float(value) for value in strikes)
    if len(values) < 2:
        return None
    diffs = [b - a for a, b in zip(values, values[1:], strict=False) if b > a]
    if not diffs:
        return None
    return float(statistics.median(diffs))


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


def _quantiles(series: pl.Series, quantiles: dict[str, float]) -> dict[str, float | None]:
    if series.is_empty():
        return {label: None for label in quantiles}
    results: dict[str, float | None] = {}
    for label, q in quantiles.items():
        try:
            value = series.quantile(q, "nearest")
        except Exception:
            value = None
        results[label] = float(value) if value is not None and not math.isnan(float(value)) else None
    return results


def _normalize_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(numeric) else numeric


def _load_quote_distance(series: str) -> tuple[float | None, str | None]:
    if not EXEC_DEFAULTS_PATH.exists():
        return None, None
    try:
        payload = json.loads(EXEC_DEFAULTS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None, None
    series_cfg = payload.get("series", {}).get(series.upper())
    if not isinstance(series_cfg, dict):
        return None, None
    for key in ("quote_distance", "quote_distance_points", "quote_distance_index_points"):
        if key in series_cfg:
            value = _safe_float(series_cfg.get(key))
            if value is not None:
                return value, f"{EXEC_DEFAULTS_PATH.name}:{key}"
    return None, None


def _build_per_window_deltas(frame: pl.DataFrame) -> list[dict[str, object]]:
    if frame.is_empty() or "window_label" not in frame.columns or "basis" not in frame.columns:
        return []
    aggregated = (
        frame.group_by("window_label")
        .agg(
            [
                pl.col("basis").count().alias("n"),
                pl.col("basis").mean().alias("mean"),
                pl.col("basis").quantile(0.05, "nearest").alias("p05"),
                pl.col("basis").quantile(0.50, "nearest").alias("p50"),
                pl.col("basis").quantile(0.95, "nearest").alias("p95"),
            ]
        )
        .sort("window_label")
    )
    rows: list[dict[str, object]] = []
    for row in aggregated.iter_rows(named=True):
        rows.append(
            {
                "window_id": row.get("window_label"),
                "n": int(row.get("n") or 0),
                "mean": _normalize_float(row.get("mean")),
                "p05": _normalize_float(row.get("p05")),
                "p50": _normalize_float(row.get("p50")),
                "p95": _normalize_float(row.get("p95")),
            }
        )
    return rows


def _compute_flip_risk(
    *,
    series: str,
    basis_series: pl.Series,
    strike_spacings: list[float],
) -> dict[str, object]:
    rationale_parts: list[str] = []
    abs_series = basis_series.abs() if not basis_series.is_empty() else pl.Series([])
    basis_abs_p95 = _normalize_float(
        abs_series.quantile(0.95, "nearest") if not abs_series.is_empty() else None
    )

    strike_spacing = _normalize_float(statistics.median(strike_spacings)) if strike_spacings else None
    quote_distance, quote_source = _load_quote_distance(series)
    default_quote_distance_used = False
    if quote_distance is None:
        quote_distance = DEFAULT_QUOTE_DISTANCE
        default_quote_distance_used = True
        rationale_parts.append(
            f"quote_distance defaulted to {DEFAULT_QUOTE_DISTANCE:.2f} (no config found)"
        )
    elif quote_source:
        rationale_parts.append(f"quote_distance sourced from {quote_source}")

    if strike_spacing is not None:
        rationale_parts.append(f"median strike spacing {strike_spacing:.2f}")
    else:
        rationale_parts.append("strike spacing unavailable (no strikes)")

    threshold_candidates: list[tuple[str, float]] = []
    if strike_spacing is not None:
        threshold_candidates.append(("strike_spacing/2", strike_spacing / 2.0))
    if quote_distance is not None:
        threshold_candidates.append(("quote_distance", quote_distance))

    threshold = None
    threshold_source = None
    if threshold_candidates:
        threshold_source, threshold = min(threshold_candidates, key=lambda item: item[1])
    else:
        rationale_parts.append("no threshold candidates available")

    if basis_abs_p95 is None:
        rationale_parts.append("basis_abs_p95 unavailable (insufficient samples)")
    else:
        rationale_parts.append(f"basis_abs_p95={basis_abs_p95:.4f}")

    if threshold is None or basis_abs_p95 is None:
        flag = True
        rationale_parts.append("flip risk flagged (fail-closed on uncertainty)")
    else:
        flag = basis_abs_p95 >= threshold
        rationale_parts.append(
            f"threshold={threshold:.4f} ({threshold_source}); flip_risk={'FAIL' if flag else 'PASS'}"
        )

    thresholds = {
        "basis_abs_p95": basis_abs_p95,
        "strike_spacing": strike_spacing,
        "quote_distance": _normalize_float(quote_distance),
        "threshold": _normalize_float(threshold),
        "threshold_source": threshold_source,
        "quantile": "p95",
        "default_quote_distance_used": default_quote_distance_used,
    }
    return {
        "flag": bool(flag),
        "rationale": "; ".join(rationale_parts),
        "thresholds": thresholds,
    }


def _build_summary(
    frame: pl.DataFrame,
    *,
    day: date,
    series: str,
) -> dict[str, object]:
    basis_series = frame.get_column("basis").drop_nulls() if "basis" in frame.columns else pl.Series([])
    basis_quantiles = _quantiles(
        basis_series,
        {
            "p01": 0.01,
            "p05": 0.05,
            "p50": 0.50,
            "p95": 0.95,
            "p99": 0.99,
        },
    )
    strike_spacings = [
        float(value)
        for value in frame.get_column("strike_spacing").drop_nulls().to_list()
        if value is not None
    ] if "strike_spacing" in frame.columns else []
    summary = {
        "series": series.upper(),
        "asof_date": day.isoformat(),
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "sample_count": int(basis_series.len()),
        "basis_quantiles": basis_quantiles,
        "per_window_deltas": _build_per_window_deltas(frame),
        "flip_risk": _compute_flip_risk(
            series=series,
            basis_series=basis_series,
            strike_spacings=strike_spacings,
        ),
    }
    return summary


def _render_report(
    frame: pl.DataFrame,
    *,
    day: date,
    series: str,
    dataset_path: Path,
    command: str,
    summary: dict[str, object],
) -> str:
    total_rows = frame.height
    basis_series = frame.get_column("basis").drop_nulls() if "basis" in frame.columns else pl.Series([])
    abs_series = basis_series.abs() if not basis_series.is_empty() else pl.Series([])
    basis_stats = _series_stats(basis_series)
    abs_stats = _series_stats(abs_series)
    basis_quantiles = summary.get("basis_quantiles", {}) if isinstance(summary, dict) else {}
    flip_summary = summary.get("flip_risk", {}) if isinstance(summary, dict) else {}
    flip_flag = bool(flip_summary.get("flag")) if isinstance(flip_summary, dict) else False
    flip_rationale = str(flip_summary.get("rationale", "")).strip() if isinstance(flip_summary, dict) else ""
    thresholds = flip_summary.get("thresholds", {}) if isinstance(flip_summary, dict) else {}
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
        "- Basis quantiles (value=Polygon-Kalshi): "
        f"p01={_format_float(_normalize_float(basis_quantiles.get('p01')))}, "
        f"p05={_format_float(_normalize_float(basis_quantiles.get('p05')))}, "
        f"p50={_format_float(_normalize_float(basis_quantiles.get('p50')))}, "
        f"p95={_format_float(_normalize_float(basis_quantiles.get('p95')))}, "
        f"p99={_format_float(_normalize_float(basis_quantiles.get('p99')))}"
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

    lines.append("## Flip-Risk Summary (daily gate)")
    lines.append(f"- Status: {'FAIL' if flip_flag else 'PASS'}")
    if thresholds:
        lines.append(
            "- Thresholds: "
            f"basis_abs_p95={_format_float(_normalize_float(thresholds.get('basis_abs_p95')))}, "
            f"strike_spacing={_format_float(_normalize_float(thresholds.get('strike_spacing')))}, "
            f"quote_distance={_format_float(_normalize_float(thresholds.get('quote_distance')))}, "
            f"threshold={_format_float(_normalize_float(thresholds.get('threshold')))} "
            f"(source={thresholds.get('threshold_source')})"
        )
    if flip_rationale:
        lines.append(f"- Rationale: {flip_rationale}")
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
    lines.append(
        "- Flip-risk flag: abs(basis) >= nearest_strike_margin when a nearest strike is available."
    )
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

    lines.append("## Per-Window Basis Deltas")
    per_window = summary.get("per_window_deltas", []) if isinstance(summary, dict) else []
    if not per_window:
        lines.append("_No per-window deltas available._")
    else:
        headers = ["window", "n", "mean", "p05", "p50", "p95"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for entry in per_window:
            lines.append(
                "| {window} | {n} | {mean} | {p05} | {p50} | {p95} |".format(
                    window=entry.get("window_id", ""),
                    n=entry.get("n", 0),
                    mean=_format_float(_normalize_float(entry.get("mean"))),
                    p05=_format_float(_normalize_float(entry.get("p05"))),
                    p50=_format_float(_normalize_float(entry.get("p50"))),
                    p95=_format_float(_normalize_float(entry.get("p95"))),
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
    if isinstance(client, _KalshiAuthenticatedClient):
        return _discover_markets_authenticated(client, day=day, series=series)
    discoveries = discover_markets_for_day(client, trading_day=day, series=[series], status=None)
    market_by_label: dict[str, DiscoveredMarket] = {}
    discovery_by_label: dict[str, WindowDiscovery] = {}
    for discovery in discoveries:
        discovery_by_label[discovery.label] = discovery
        for market in discovery.markets:
            if market.series.upper() == series.upper():
                market_by_label[discovery.label] = market
    return market_by_label, discovery_by_label


def _discover_markets_authenticated(
    client: _KalshiAuthenticatedClient,
    *,
    day: date,
    series: str,
) -> tuple[dict[str, DiscoveredMarket], dict[str, WindowDiscovery]]:
    windows = _windows_for_series(day, series)
    if not windows:
        return {}, {}

    events: list[dict[str, Any]] = []
    seen: set[str] = set()
    for status in ("settled", "closed", "open"):
        try:
            payloads = client.list_events(series_ticker=series, status=status)
        except KalshiHttpError:
            continue
        for payload in payloads:
            event_ticker = str(payload.get("event_ticker") or "").strip()
            if not event_ticker or event_ticker in seen:
                continue
            seen.add(event_ticker)
            events.append(payload)

    discovered: list[DiscoveredMarket] = []
    for payload in events:
        event_ticker = str(payload.get("event_ticker") or "").strip()
        if not event_ticker:
            continue
        close_utc = _parse_iso(
            payload.get("strike_date")
            or payload.get("expiration_time")
            or payload.get("close_time")
        )
        if close_utc is None:
            close_utc = _close_time_from_ticker(event_ticker)
        if close_utc is None:
            continue
        if close_utc.astimezone(ET).date() != day:
            continue
        status = payload.get("status")
        discovered.append(
            DiscoveredMarket(
                series=series.upper(),
                event_id=event_ticker,
                event_ticker=event_ticker,
                close_time=close_utc,
                market_ids=(),
                market_tickers=(),
                market_count=0,
                status=str(status).lower() if isinstance(status, str) else None,
            )
        )

    market_by_label: dict[str, DiscoveredMarket] = {}
    for window in windows:
        best: DiscoveredMarket | None = None
        best_delta: float | None = None
        for market in discovered:
            delta = abs((market.close_time_et - window.target_et).total_seconds())
            if delta > EVENT_MATCH_TOLERANCE_SECONDS:
                continue
            if best_delta is None or delta < best_delta:
                best = market
                best_delta = delta
        if best is not None:
            market_by_label[window.label] = best

    return market_by_label, {}


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
        kalshi_client = _KalshiAuthenticatedClient()
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
        strike_spacing = _median_spacing(strikes)
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
                "strike_spacing": strike_spacing,
                "flip_risk": flip_risk,
            }
        )

    frame = pl.DataFrame(rows)
    return frame


def _command_for_series(raw_args: Sequence[str], series: str, *, use_all_series: bool) -> str:
    parts: list[str] = ["python", "tools/settlement_basis_audit.py"]
    if use_all_series:
        filtered: list[str] = []
        skip_next = False
        for arg in raw_args:
            if skip_next:
                skip_next = False
                continue
            if arg == "--all-series":
                continue
            if arg == "--series":
                skip_next = True
                continue
            filtered.append(str(arg))
        filtered.extend(["--series", series])
        parts.extend(filtered)
    else:
        parts.extend(str(arg) for arg in raw_args)
    return " ".join(shlex.quote(part) for part in parts)


def _run_audit_for_series(
    *,
    day: date,
    series: str,
    out_report: str | None,
    out_data: str | None,
    out_json: str | None,
    use_cache: bool,
    offline_root: Path | None,
    command: str,
    archive_dir: Path | None,
) -> tuple[Path, Path, Path]:
    report_path, data_path, json_path = _resolve_paths(
        day,
        series,
        out_report=out_report,
        out_data=out_data,
        out_json=out_json,
    )

    if use_cache:
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found for --use-cache: {data_path}")
        frame = _load_frame(data_path)
        summary = _build_summary(frame, day=day, series=series)
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        report = _render_report(
            frame,
            day=day,
            series=series,
            dataset_path=data_path,
            command=command,
            summary=summary,
        )
        report_path.write_text(report, encoding="utf-8")
        print(f"[settlement_basis] summary written to {json_path}")
        print(f"[settlement_basis] report written to {report_path}")
        if archive_dir is not None:
            _archive_outputs(archive_dir, [json_path, report_path])
        return report_path, data_path, json_path

    frame = _build_dataset(day=day, series=series, offline_fixtures=offline_root)
    _write_frame(frame, data_path)
    summary = _build_summary(frame, day=day, series=series)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = _render_report(
        frame,
        day=day,
        series=series,
        dataset_path=data_path,
        command=command,
        summary=summary,
    )
    report_path.write_text(report, encoding="utf-8")
    print(f"[settlement_basis] dataset written to {data_path}")
    print(f"[settlement_basis] summary written to {json_path}")
    print(f"[settlement_basis] report written to {report_path}")
    if archive_dir is not None:
        _archive_outputs(archive_dir, [json_path, report_path])
    return report_path, data_path, json_path


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    raw_args = argv if argv is not None else sys.argv[1:]
    archive_dir = _resolve_archive_dir(args.archive_dir, args.runlog)

    if args.all_series:
        if args.out_report or args.out_data or args.out_json:
            raise ValueError("--out-report/--out-data/--out-json are not supported with --all-series.")
        offline_root = Path(args.offline_fixtures).resolve() if args.offline_fixtures else None
        for series in SUPPORTED_SERIES_ORDER:
            command = _command_for_series(raw_args, series, use_all_series=True)
            _run_audit_for_series(
                day=args.day,
                series=series,
                out_report=None,
                out_data=None,
                out_json=None,
                use_cache=args.use_cache,
                offline_root=offline_root,
                command=command,
                archive_dir=archive_dir,
            )
        return

    series = str(args.series)
    command = _command_for_series(raw_args, series, use_all_series=False)
    offline_root = Path(args.offline_fixtures).resolve() if args.offline_fixtures else None
    _run_audit_for_series(
        day=args.day,
        series=series,
        out_report=args.out_report,
        out_data=args.out_data,
        out_json=args.out_json,
        use_cache=args.use_cache,
        offline_root=offline_root,
        command=command,
        archive_dir=archive_dir,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
