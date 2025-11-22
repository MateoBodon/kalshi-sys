"""GO/NO-GO checks for SPX/NDX index ladder windows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import os
from pathlib import Path
from typing import Callable, Iterable

import requests
from zoneinfo import ZoneInfo

from kalshi_alpha.datastore.paths import PROC_ROOT
from kalshi_alpha.exec.heartbeat import kill_switch_engaged, resolve_kill_switch_path
from kalshi_alpha.strategies.index.model_polygon import PARAM_ROOT, params_path
from kalshi_alpha.utils.env import load_env
from kalshi_alpha.utils.keys import load_polygon_api_key

ET = ZoneInfo("America/New_York")

SERIES_HORIZONS: tuple[tuple[str, str], ...] = (
    ("INX", "close"),
    ("NASDAQ100", "close"),
    ("INXU", "noon"),
    ("NASDAQ100U", "noon"),
)

MAX_CALIBRATION_AGE_DAYS = 14.0
POLYGON_PING_URL = "https://api.polygon.io/v1/marketstatus/now"


@dataclass(slots=True)
class PreflightResult:
    go: bool
    reasons: list[str]
    details: dict[str, object]

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return self.go


def _ensure_et(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC).astimezone(ET)
    return moment.astimezone(ET)


def _file_age_days(path: Path, now: datetime) -> float | None:
    """Return age in days using generated_at when present, else mtime."""

    try:
        payload = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return None

    generated_at = None
    if payload:
        import json

        try:
            parsed = json.loads(payload)
            generated_at = parsed.get("generated_at")
        except json.JSONDecodeError:
            generated_at = None
    if isinstance(generated_at, str) and generated_at:
        try:
            timestamp = datetime.fromisoformat(generated_at)
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)
            age_seconds = (now.astimezone(UTC) - timestamp.astimezone(UTC)).total_seconds()
            return max(age_seconds, 0.0) / 86400.0
        except ValueError:
            generated_at = None

    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    except OSError:
        return None
    age_seconds = (now.astimezone(UTC) - mtime).total_seconds()
    return max(age_seconds, 0.0) / 86400.0


def _calibration_check(
    *,
    now: datetime,
    params_root: Path,
    max_age_days: float,
) -> tuple[bool, list[str], dict[str, float]]:
    reasons: list[str] = []
    ages: dict[str, float] = {}
    for series, horizon in SERIES_HORIZONS:
        path = params_path(series, horizon, root=params_root)
        if not path.exists():
            reasons.append(f"calibration_missing:{series}:{horizon}")
            continue
        age = _file_age_days(path, now)
        if age is not None:
            ages[f"{series}:{horizon}"] = age
            if age > max_age_days:
                reasons.append(f"calibration_stale:{series}:{horizon}:{age:.1f}d")
    return not reasons, reasons, ages


def _polygon_ping(timeout: float) -> bool:
    api_key = load_polygon_api_key()
    if not api_key:
        return False
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    try:
        response = requests.get(POLYGON_PING_URL, headers=headers, timeout=timeout)
    except requests.RequestException:
        return False
    return response.status_code < 400


def _missing_env_vars(vars_to_check: Iterable[str]) -> list[str]:
    missing: list[str] = []
    for key in vars_to_check:
        if not os.getenv(key, "").strip():
            missing.append(key)
    return missing


def run_preflight(
    now_et: datetime,
    *,
    params_root: Path | None = None,
    kill_switch_file: Path | None = None,
    polygon_timeout: float = 2.0,
    polygon_ping: Callable[[float], bool] | None = None,
    require_kalshi: bool = True,
) -> PreflightResult:
    """Evaluate GO/NO-GO checks for index ladder windows."""

    load_env()
    reasons: list[str] = []
    details: dict[str, object] = {}

    reference_et = _ensure_et(now_et)
    now_utc = reference_et.astimezone(UTC)

    # Environment + secrets -------------------------------------------------
    env_missing: list[str] = []
    if require_kalshi:
        env_missing.extend(_missing_env_vars(["KALSHI_API_KEY_ID", "KALSHI_PRIVATE_KEY_PEM_PATH"]))
        key_path_raw = os.getenv("KALSHI_PRIVATE_KEY_PEM_PATH", "").strip()
        if key_path_raw:
            key_path = Path(key_path_raw).expanduser()
            if not key_path.exists():
                reasons.append("kalshi_private_key_missing")
                details["kalshi_private_key_path"] = key_path.as_posix()
    polygon_key_present = bool(load_polygon_api_key())
    if not polygon_key_present:
        env_missing.append("POLYGON_API_KEY")
    if env_missing:
        reasons.append("missing_env:" + ",".join(sorted(env_missing)))
    details["env_missing"] = sorted(env_missing)

    # Kill switch -----------------------------------------------------------
    kill_switch_path = resolve_kill_switch_path(kill_switch_file)
    if kill_switch_engaged(kill_switch_path):
        reasons.append("kill_switch_engaged")
        details["kill_switch_path"] = kill_switch_path.as_posix()

    # Calibration freshness -------------------------------------------------
    params_root_resolved = Path(params_root or PARAM_ROOT)
    params_root_resolved.mkdir(parents=True, exist_ok=True)
    _, calib_reasons, calib_ages = _calibration_check(
        now=now_utc,
        params_root=params_root_resolved,
        max_age_days=MAX_CALIBRATION_AGE_DAYS,
    )
    if calib_reasons:
        reasons.extend(calib_reasons)
    details["calibration_age_days"] = calib_ages

    # Polygon connectivity --------------------------------------------------
    if polygon_key_present:
        ping_fn = polygon_ping or _polygon_ping
        if not ping_fn(polygon_timeout):
            reasons.append("polygon_unreachable")

    go = not reasons
    details["evaluated_at_et"] = reference_et.isoformat()
    return PreflightResult(go=go, reasons=reasons, details=details)


__all__ = ["MAX_CALIBRATION_AGE_DAYS", "PreflightResult", "run_preflight"]
