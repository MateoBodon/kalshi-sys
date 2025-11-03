"""Data feed freshness monitor for ramp readiness."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from kalshi_alpha.datastore import paths as datastore_paths
from kalshi_alpha.exec.monitors.summary import MONITOR_ARTIFACTS_DIR

FRESHNESS_CONFIG_PATH = Path("configs/freshness.yaml")
FRESHNESS_ARTIFACT_PATH = MONITOR_ARTIFACTS_DIR / "freshness.json"


@dataclass(slots=True)
class FeedState:
    """Computed freshness payload for a single feed."""

    feed_id: str
    label: str
    required: bool
    last_ts: datetime | None
    age_minutes: float | None
    ok: bool
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_metrics(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.feed_id,
            "label": self.label,
            "required": self.required,
            "ok": self.ok,
            "age_minutes": self._round(self.age_minutes),
            "last_ts": self.last_ts.isoformat() if self.last_ts else None,
        }
        if self.reason:
            payload["reason"] = self.reason
        if self.details:
            payload["details"] = _json_safe(self.details)
        return payload

    @staticmethod
    def _round(value: float | None) -> float | None:
        if value is None:
            return None
        return round(value, 3)


@dataclass(slots=True)
class FreshnessConfig:
    feeds: dict[str, dict[str, Any]]
    required_order: list[str]


def load_config(path: Path | None = None) -> FreshnessConfig:
    """Load freshness configuration and merge with defaults."""

    defaults: dict[str, dict[str, Any]] = {
        "bls_cpi.latest_release": {
            "label": "BLS CPI (latest release)",
            "age_days": 35.0,
            "required": True,
            "namespace": ("bls_cpi", "latest_release"),
            "timestamp_column": "release_datetime",
        },
        "dol_claims.latest_report": {
            "label": "DOL Claims (ETA-539)",
            "age_days": 8.0,
            "required": True,
            "namespace": ("dol_claims", "latest_report"),
            "timestamp_column": "week_ending",
        },
        "treasury_10y.daily": {
            "label": "Treasury 10Y Par Yield",
            "age_business_days": 3.0,
            "required": True,
            "namespace": ("treasury_yields", "daily"),
            "timestamp_column": "as_of",
            "expected_maturity": "DGS10",
        },
        "cleveland_nowcast.monthly": {
            "label": "Cleveland Fed Nowcast (monthly)",
            "age_days": 35.0,
            "required": True,
            "namespace": ("cleveland_nowcast", "monthly"),
            "timestamp_column": "as_of",
        },
        "aaa_gas.daily": {
            "label": "AAA National Gas (daily)",
            "age_days": 2.0,
            "required": True,
            "price_min": 2.0,
            "price_max": 6.0,
        },
        "nws_daily_climate": {
            "label": "NWS Daily Climate (stations)",
            "age_days": 2.0,
            "required": True,
            "active_stations": [],
            "namespace": ("nws_cli", "daily_climate"),
        },
        "polygon_index.websocket": {
            "label": "Polygon index websocket",
            "age_seconds": 2.0,
            "required": True,
            "namespace": "polygon_index",
        },
    }

    raw_config: Mapping[str, Any] = {}
    config_path = path or FRESHNESS_CONFIG_PATH
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            raw_config = loaded

    feeds_cfg = raw_config.get("feeds") if isinstance(raw_config, dict) else {}
    merged: dict[str, dict[str, Any]] = {}
    for feed_id, default in defaults.items():
        overrides = feeds_cfg.get(feed_id) if isinstance(feeds_cfg, dict) else {}
        merged_config = {**default}
        if isinstance(overrides, dict):
            for key, value in overrides.items():
                merged_config[key] = value
        merged[feed_id] = merged_config

    if isinstance(feeds_cfg, dict):
        for feed_id, extra_cfg in feeds_cfg.items():
            if feed_id in merged or not isinstance(extra_cfg, dict):
                continue
            merged[feed_id] = {**extra_cfg}

    required_order = raw_config.get("required_order")
    if not isinstance(required_order, list):
        required_order = list(defaults.keys())

    return FreshnessConfig(feeds=merged, required_order=[str(item) for item in required_order])


def compute_freshness(
    *,
    config: FreshnessConfig | None = None,
    now: datetime | None = None,
    proc_root: Path | None = None,
) -> tuple[list[FeedState], dict[str, Any]]:
    """Compute per-feed freshness state and aggregate metrics."""

    cfg = config or load_config()
    moment = _ensure_utc(now or datetime.now(tz=UTC))
    base_proc = proc_root or datastore_paths.PROC_ROOT
    base_raw = datastore_paths.RAW_ROOT if proc_root is None else proc_root.parent / "raw"

    feed_states: list[FeedState] = []
    for feed_id in cfg.required_order:
        feed_cfg = cfg.feeds.get(feed_id)
        if not isinstance(feed_cfg, dict):
            continue
        feed_state = _evaluate_feed(feed_id, feed_cfg, moment, base_proc, base_raw)
        feed_states.append(feed_state)

    required_feeds = [state for state in feed_states if state.required]
    stale_required = [state for state in required_feeds if not state.ok]
    required_ok = not stale_required
    metrics = {
        "generated_at": moment.isoformat(),
        "required_feeds_ok": required_ok,
        "required_feeds": [state.feed_id for state in required_feeds],
        "stale_feeds": [state.feed_id for state in stale_required],
        "feeds": [state.to_metrics() for state in feed_states],
    }
    return feed_states, metrics


def write_freshness_artifact(
    *,
    config_path: Path | None = None,
    output_path: Path | None = None,
    now: datetime | None = None,
    proc_root: Path | None = None,
    emit_table: bool = True,
) -> dict[str, Any]:
    """Compute freshness metrics and persist monitor artifact."""

    cfg = load_config(config_path)
    feed_states, metrics = compute_freshness(config=cfg, now=now, proc_root=proc_root)
    if emit_table:
        metrics["table"] = _build_table(feed_states)

    required_ok = bool(metrics.get("required_feeds_ok", True))
    stale = metrics.get("stale_feeds") or []
    if stale:
        summary = "Stale feeds: " + ", ".join(stale)
    else:
        summary = "All required feeds fresh"

    payload = {
        "name": "data_freshness",
        "status": "OK" if required_ok else "ALERT",
        "generated_at": metrics["generated_at"],
        "message": summary,
        "metrics": metrics,
    }

    target_path = output_path or FRESHNESS_ARTIFACT_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def load_artifact(path: Path | None = None) -> dict[str, Any] | None:
    """Load freshness artifact from disk."""

    artifact_path = path or FRESHNESS_ARTIFACT_PATH
    if not artifact_path.exists():
        return None
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def summarize_artifact(payload: dict[str, Any] | None, *, artifact_path: Path) -> dict[str, Any]:
    """Normalize the freshness artifact into a ramp-friendly summary."""

    summary: dict[str, Any] = {
        "status": "MISSING",
        "artifact_path": artifact_path.as_posix(),
        "required_feeds_ok": False,
        "required_feeds": [],
        "stale_feeds": [],
        "feeds": [],
        "generated_at": None,
    }
    if not isinstance(payload, dict):
        summary["message"] = "data freshness artifact missing"
        return summary

    status = str(payload.get("status") or "UNKNOWN").upper()
    summary["status"] = status
    generated_at = payload.get("generated_at")
    if isinstance(generated_at, str):
        summary["generated_at"] = generated_at
    message = payload.get("message")
    if isinstance(message, str):
        summary["message"] = message
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        required_ok = metrics.get("required_feeds_ok")
        summary["required_feeds_ok"] = bool(required_ok) if isinstance(required_ok, bool) else bool(required_ok)
        required = metrics.get("required_feeds") or []
        if isinstance(required, list):
            summary["required_feeds"] = [str(item) for item in required]
        stale = metrics.get("stale_feeds") or []
        if isinstance(stale, list):
            summary["stale_feeds"] = [str(item) for item in stale]
        feeds = metrics.get("feeds") or metrics.get("table") or []
        if isinstance(feeds, list):
            summary["feeds"] = [dict(entry) for entry in feeds if isinstance(entry, dict)]
    return summary


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute data freshness monitors.")
    parser.add_argument("--config", type=Path, default=None, help="Optional freshness config override.")
    parser.add_argument("--proc-root", type=Path, default=None, help="Override processed data root.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Custom output path (default: reports/_artifacts/monitors/freshness.json).",
    )
    parser.add_argument("--print", action="store_true", help="Print freshness table to stdout.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    payload = write_freshness_artifact(
        config_path=args.config,
        output_path=args.output,
        proc_root=args.proc_root,
    )
    if args.print:
        metrics = payload.get("metrics", {})
        table = metrics.get("table") or []
        print("Data Freshness")
        print("--------------")
        for row in table:
            label = row.get("label") or row.get("id")
            required = "yes" if row.get("required") else "no"
            ok_state = "OK" if row.get("ok") else "STALE"
            age = row.get("age_minutes")
            reason = row.get("reason")
            age_str = f"{age:.2f} min" if isinstance(age, (int, float)) else "n/a"
            suffix = f" ({reason})" if reason else ""
            print(f"- {label}: {ok_state} | required={required} | age={age_str}{suffix}")
        print("")
        status_line = "GO" if payload.get("status") == "OK" else "NO-GO"
        print(f"Decision: {status_line}")
    return 0


def _evaluate_feed(
    feed_id: str,
    cfg: Mapping[str, Any],
    now: datetime,
    proc_root: Path,
    raw_root: Path,
) -> FeedState:
    label = str(cfg.get("label") or feed_id)
    required = bool(cfg.get("required", True))
    handlers: dict[str, Callable[[Mapping[str, Any], str, bool, datetime, Path], FeedState]] = {
        "bls_cpi.latest_release": _evaluate_bls,
        "dol_claims.latest_report": _evaluate_claims,
        "treasury_10y.daily": _evaluate_treasury,
        "cleveland_nowcast.monthly": _evaluate_cleveland,
        "aaa_gas.daily": _evaluate_aaa,
        "nws_daily_climate": _evaluate_weather,
        "polygon_index.websocket": lambda cfg_item, feed_label, required_flag, current_time, _: _evaluate_polygon_ws(
            cfg_item,
            feed_label,
            required_flag,
            current_time,
            raw_root,
        ),
    }
    handler = handlers.get(feed_id)
    if handler is None:
        return FeedState(feed_id, label, required, None, None, not required, "UNKNOWN_FEED")
    return handler(cfg, label, required, now, proc_root)


def _evaluate_polygon_ws(
    cfg: Mapping[str, Any],
    label: str,
    required: bool,
    now: datetime,
    raw_root: Path,
) -> FeedState:
    namespace = str(cfg.get("namespace", "polygon_index"))
    threshold_seconds = _to_float(cfg.get("age_seconds"), default=2.0)
    latest_path = _latest_snapshot_path(raw_root, namespace)
    if latest_path is None:
        return FeedState("polygon_index.websocket", label, required, None, None, not required, "NO_SNAPSHOTS")
    last_ts = _parse_snapshot_timestamp(latest_path.name)
    if last_ts is None:
        return FeedState("polygon_index.websocket", label, required, None, None, False, "BAD_TIMESTAMP")
    age_seconds = max((now - last_ts).total_seconds(), 0.0)
    age_minutes = age_seconds / 60.0
    ok = True
    if threshold_seconds is not None and age_seconds > threshold_seconds:
        ok = False
    reason = None if ok else f"STALE>{threshold_seconds}s"
    details = {
        "age_seconds": round(age_seconds, 3),
        "threshold_seconds": threshold_seconds,
        "snapshot_path": str(latest_path),
    }
    return FeedState(
        "polygon_index.websocket",
        label,
        required,
        last_ts,
        age_minutes,
        ok or not required,
        reason,
        details,
    )


def _evaluate_bls(
    cfg: Mapping[str, Any],
    label: str,
    required: bool,
    now: datetime,
    proc_root: Path,
) -> FeedState:
    namespace = cfg.get("namespace", ("bls_cpi", "latest_release"))
    timestamp_column = str(cfg.get("timestamp_column", "release_datetime"))
    threshold_days = _to_float(cfg.get("age_days"), default=35.0)
    path = _latest_parquet(proc_root, namespace)
    if path is None:
        return FeedState("bls_cpi.latest_release", label, required, None, None, not required, "MISSING")
    frame = pl.read_parquet(path)
    if timestamp_column not in frame.columns:
        return FeedState("bls_cpi.latest_release", label, required, None, None, False, "MISSING_TIMESTAMP")
    last_ts = _extract_datetime(frame.select(pl.col(timestamp_column).max()).item())
    if last_ts is None:
        return FeedState("bls_cpi.latest_release", label, required, None, None, False, "NO_DATA")
    age_minutes = _age_minutes(last_ts, now)
    age_days = _age_days(last_ts, now)
    ok = True
    if threshold_days is not None and age_days is not None and age_days > threshold_days:
        ok = False
    reason = None if ok else f"STALE>{threshold_days}d"
    details = {
        "age_days": _round(age_days),
        "threshold_days": threshold_days,
        "count": frame.height,
    }
    return FeedState(
        "bls_cpi.latest_release",
        label,
        required,
        last_ts,
        age_minutes,
        ok or not required,
        reason,
        details,
    )


def _evaluate_claims(
    cfg: Mapping[str, Any],
    label: str,
    required: bool,
    now: datetime,
    proc_root: Path,
) -> FeedState:
    namespace = cfg.get("namespace", ("dol_claims", "latest_report"))
    timestamp_column = str(cfg.get("timestamp_column", "week_ending"))
    threshold_days = _to_float(cfg.get("age_days"), default=8.0)
    path = _latest_parquet(proc_root, namespace)
    if path is None:
        return FeedState("dol_claims.latest_report", label, required, None, None, not required, "MISSING")
    frame = pl.read_parquet(path)
    if timestamp_column not in frame.columns:
        return FeedState("dol_claims.latest_report", label, required, None, None, False, "MISSING_TIMESTAMP")
    last_ts = _extract_date_as_datetime(frame.select(pl.col(timestamp_column).max()).item())
    if last_ts is None:
        return FeedState("dol_claims.latest_report", label, required, None, None, False, "NO_DATA")
    age_minutes = _age_minutes(last_ts, now)
    age_days = _age_days(last_ts, now)
    ok = True
    if threshold_days is not None and age_days is not None and age_days > threshold_days:
        ok = False
    reason = None if ok else f"STALE>{threshold_days}d"
    details = {"age_days": _round(age_days), "threshold_days": threshold_days, "count": frame.height}
    return FeedState(
        "dol_claims.latest_report",
        label,
        required,
        last_ts,
        age_minutes,
        ok or not required,
        reason,
        details,
    )


def _evaluate_treasury(
    cfg: Mapping[str, Any],
    label: str,
    required: bool,
    now: datetime,
    proc_root: Path,
) -> FeedState:
    namespace = cfg.get("namespace", ("treasury_yields", "daily"))
    timestamp_column = str(cfg.get("timestamp_column", "as_of"))
    expected_maturity = str(cfg.get("expected_maturity", "DGS10")).upper()
    threshold_business_days = _to_float(cfg.get("age_business_days"), default=3.0)
    path = _latest_parquet(proc_root, namespace)
    if path is None:
        return FeedState("treasury_10y.daily", label, required, None, None, not required, "MISSING")
    frame = pl.read_parquet(path)
    if timestamp_column not in frame.columns or "maturity" not in frame.columns:
        return FeedState("treasury_10y.daily", label, required, None, None, False, "MISSING_COLUMNS")
    normalized = frame.with_columns(pl.col("maturity").str.to_uppercase())
    subset = normalized.filter(pl.col("maturity") == expected_maturity)
    if subset.is_empty():
        return FeedState("treasury_10y.daily", label, required, None, None, False, "TENY_SERIES_MISMATCH")
    last_ts = _extract_date_as_datetime(subset.select(pl.col(timestamp_column).max()).item())
    if last_ts is None:
        return FeedState("treasury_10y.daily", label, required, None, None, False, "NO_DATA")
    age_minutes = _age_minutes(last_ts, now)
    age_business = _business_days(last_ts.date(), now.date())
    ok = True
    if threshold_business_days is not None and age_business > threshold_business_days:
        ok = False
    reason = None if ok else f"STALE>{threshold_business_days}bd"
    details = {
        "age_business_days": _round(age_business),
        "threshold_business_days": threshold_business_days,
        "count": int(subset.height),
    }
    return FeedState(
        "treasury_10y.daily",
        label,
        required,
        last_ts,
        age_minutes,
        ok or not required,
        reason,
        details,
    )


def _evaluate_cleveland(
    cfg: Mapping[str, Any],
    label: str,
    required: bool,
    now: datetime,
    proc_root: Path,
) -> FeedState:
    namespace = cfg.get("namespace", ("cleveland_nowcast", "monthly"))
    timestamp_column = str(cfg.get("timestamp_column", "as_of"))
    threshold_days = _to_float(cfg.get("age_days"), default=35.0)
    path = _latest_parquet(proc_root, namespace)
    if path is None:
        return FeedState("cleveland_nowcast.monthly", label, required, None, None, not required, "MISSING")
    frame = pl.read_parquet(path)
    if timestamp_column not in frame.columns:
        return FeedState("cleveland_nowcast.monthly", label, required, None, None, False, "MISSING_TIMESTAMP")
    last_ts = _extract_datetime(frame.select(pl.col(timestamp_column).max()).item())
    if last_ts is None:
        return FeedState("cleveland_nowcast.monthly", label, required, None, None, False, "NO_DATA")
    age_minutes = _age_minutes(last_ts, now)
    age_days = _age_days(last_ts, now)
    ok = True
    if threshold_days is not None and age_days is not None and age_days > threshold_days:
        ok = False
    reason = None if ok else f"STALE>{threshold_days}d"
    details = {"age_days": _round(age_days), "threshold_days": threshold_days, "count": frame.height}
    return FeedState(
        "cleveland_nowcast.monthly",
        label,
        required,
        last_ts,
        age_minutes,
        ok or not required,
        reason,
        details,
    )


def _evaluate_aaa(
    cfg: Mapping[str, Any],
    label: str,
    required: bool,
    now: datetime,
    proc_root: Path,
) -> FeedState:
    threshold_days = _to_float(cfg.get("age_days"), default=2.0)
    price_min = _to_float(cfg.get("price_min"), default=2.0)
    price_max = _to_float(cfg.get("price_max"), default=6.0)
    daily_path = proc_root / "aaa_daily.parquet"
    if not daily_path.exists():
        return FeedState("aaa_gas.daily", label, required, None, None, not required, "MISSING")
    frame = pl.read_parquet(daily_path)
    if frame.is_empty() or "date" not in frame.columns or "price" not in frame.columns:
        return FeedState("aaa_gas.daily", label, required, None, None, False, "NO_DATA")
    idx = frame.select(pl.col("date").max()).to_series().item()
    last_ts = _extract_date_as_datetime(idx)
    if last_ts is None:
        return FeedState("aaa_gas.daily", label, required, None, None, False, "INVALID_DATE")
    latest_row = frame.filter(pl.col("date") == idx).sort("date").tail(1)
    price = float(latest_row["price"][0]) if not latest_row.is_empty() else None
    age_minutes = _age_minutes(last_ts, now)
    age_days = _age_days(last_ts, now)
    stale = threshold_days is not None and age_days is not None and age_days > threshold_days
    out_of_range = (
        price is None
        or (price_min is not None and price < price_min)
        or (price_max is not None and price > price_max)
    )
    ok = (not stale) and (not out_of_range)
    reason = None
    if out_of_range:
        reason = "AAA_OUT_OF_RANGE"
    elif stale:
        reason = f"STALE>{threshold_days}d"
    details = {
        "age_days": _round(age_days),
        "threshold_days": threshold_days,
        "price": price,
        "price_min": price_min,
        "price_max": price_max,
    }
    return FeedState(
        "aaa_gas.daily",
        label,
        required,
        last_ts,
        age_minutes,
        ok or not required,
        reason,
        details,
    )


def _evaluate_weather(
    cfg: Mapping[str, Any],
    label: str,
    required: bool,
    now: datetime,
    proc_root: Path,
) -> FeedState:
    namespace = cfg.get("namespace", ("nws_cli", "daily_climate"))
    threshold_days = _to_float(cfg.get("age_days"), default=2.0)
    active_stations_raw = cfg.get("active_stations") or []
    active_stations = sorted({str(item).upper() for item in active_stations_raw if str(item).strip()})
    if not active_stations:
        not_required_details = {"stations": [], "note": "not_required"}
        return FeedState("nws_daily_climate", label, False, None, None, True, "NOT_REQUIRED", not_required_details)
    path = _latest_parquet(proc_root, namespace)
    if path is None:
        return FeedState("nws_daily_climate", label, required, None, None, False, "MISSING")
    frame = pl.read_parquet(path)
    if frame.is_empty() or "station_id" not in frame.columns or "record_date" not in frame.columns:
        return FeedState("nws_daily_climate", label, required, None, None, False, "MISSING_COLUMNS")
    frame = frame.with_columns(pl.col("station_id").str.to_uppercase())
    station_entries: list[dict[str, Any]] = []
    stale_stations: list[str] = []
    worst_age: float | None = None
    last_ts_overall: datetime | None = None
    for station in active_stations:
        subset = frame.filter(pl.col("station_id") == station)
        if subset.is_empty():
            stale_stations.append(station)
            station_entries.append(
                {
                    "station": station,
                    "ok": False,
                    "reason": "MISSING",
                    "last_ts": None,
                    "age_days": None,
                }
            )
            continue
        value = subset.select(pl.col("record_date").max()).item()
        station_ts = _extract_date_as_datetime(value)
        if station_ts is None:
            stale_stations.append(station)
            station_entries.append(
                {
                    "station": station,
                    "ok": False,
                    "reason": "INVALID_DATE",
                    "last_ts": None,
                    "age_days": None,
                }
            )
            continue
        age_days = _age_days(station_ts, now)
        age_minutes = _age_minutes(station_ts, now)
        if age_minutes is None or age_days is None:
            stale_stations.append(station)
            station_entries.append(
                {
                    "station": station,
                    "ok": False,
                    "reason": "AGE_UNKNOWN",
                    "last_ts": station_ts.isoformat(),
                    "age_days": None,
                    "age_minutes": None,
                    "threshold_days": threshold_days,
                }
            )
            continue
        if last_ts_overall is None or station_ts > last_ts_overall:
            last_ts_overall = station_ts
        if worst_age is None or age_minutes > worst_age:
            worst_age = age_minutes
        station_ok = threshold_days is None or age_days <= threshold_days
        if not station_ok:
            stale_stations.append(station)
        station_entries.append(
            {
                "station": station,
                "ok": station_ok,
                "age_days": _round(age_days),
                "age_minutes": _round(age_minutes),
                "last_ts": station_ts.isoformat(),
                "threshold_days": threshold_days,
            }
        )
    ok = not stale_stations
    reason = None if ok else "STALE_STATIONS:" + ",".join(sorted(stale_stations))
    details: dict[str, Any] = {
        "stations": station_entries,
        "threshold_days": threshold_days,
        "active_station_count": len(active_stations),
    }
    return FeedState(
        "nws_daily_climate",
        label,
        required,
        last_ts_overall,
        worst_age,
        ok or not required,
        reason,
        details,
    )


def _latest_parquet(proc_root: Path, namespace: Iterable[str | Path]) -> Path | None:
    directory = proc_root.joinpath(*namespace)
    if not directory.exists():
        return None
    files = sorted(directory.glob("*.parquet"))
    return files[-1] if files else None


def _latest_snapshot_path(raw_root: Path, namespace: str) -> Path | None:
    base = raw_root if raw_root.exists() else datastore_paths.RAW_ROOT
    pattern = base.glob(f"*/*/*/{namespace}/*.json")
    files = list(pattern)
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_mtime)


def _parse_snapshot_timestamp(name: str) -> datetime | None:
    prefix = name.split("_", 1)[0]
    try:
        parsed = datetime.strptime(prefix, "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
    except ValueError:
        return None
    return parsed


def _extract_datetime(value: datetime | date | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _ensure_utc(value)
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time(), tzinfo=UTC)
    return None


def _extract_date_as_datetime(value: datetime | date | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _ensure_utc(value)
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time(), tzinfo=UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        return _ensure_utc(parsed)
    return None


def _ensure_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def _age_minutes(last_ts: datetime | None, now: datetime) -> float | None:
    if last_ts is None:
        return None
    delta = now - last_ts
    if delta.total_seconds() < 0:
        return 0.0
    return delta.total_seconds() / 60.0


def _age_days(last_ts: datetime | None, now: datetime) -> float | None:
    age_minutes = _age_minutes(last_ts, now)
    if age_minutes is None:
        return None
    return age_minutes / (60.0 * 24.0)


def _business_days(start: date, end: date) -> float:
    if start > end:
        return 0.0
    current = start
    days = 0.0
    while current < end:
        if current.weekday() < 5:
            days += 1.0
        current += timedelta(days=1)
    return days


def _to_float(value: object, *, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 3)


def _build_table(states: Iterable[FeedState]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state in states:
        row = state.to_metrics()
        rows.append(row)
    return rows


def _json_safe(data: object) -> object:
    if isinstance(data, dict):
        return {str(key): _json_safe(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_json_safe(item) for item in data]
    if isinstance(data, datetime):
        return data.isoformat()
    if isinstance(data, (float, int, str)) or data is None:
        return data
    if isinstance(data, date):
        return data.isoformat()
    return str(data)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
