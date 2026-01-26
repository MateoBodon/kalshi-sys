"""GO/NO-GO checks for SPX/NDX index ladder windows."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
import json
import os
from pathlib import Path
from typing import Callable, Iterable, Sequence

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
GO_NO_GO_PATH = Path("reports/_artifacts/go_no_go.json")
FRESHNESS_SCOPE = "index"
BASIS_AUDIT_ROOT = PROC_ROOT / "basis"


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


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    raw = str(value).strip()
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


def _basis_audit_path(root: Path, series: str, asof: date) -> Path:
    return root / series.upper() / f"{asof.isoformat()}.json"


def _check_basis_audit(
    *,
    series: str,
    asof: date,
    now_utc: datetime,
    root: Path,
) -> tuple[bool, list[str], dict[str, object]]:
    reasons: list[str] = []
    details: dict[str, object] = {}
    path = _basis_audit_path(root, series, asof)
    details["path"] = path.as_posix()
    if not path.exists():
        reasons.append(f"basis_audit_missing:{series}")
        details["status"] = "MISSING"
        return False, reasons, details
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        reasons.append(f"basis_audit_stale:{series}:unreadable")
        details["status"] = "STALE"
        details["error"] = str(exc)
        return False, reasons, details

    asof_payload = payload.get("asof_date")
    details["asof_date"] = asof_payload
    if asof_payload != asof.isoformat():
        reasons.append(f"basis_audit_stale:{series}:asof_mismatch")

    generated_at_raw = payload.get("generated_at")
    generated_at = _parse_timestamp(str(generated_at_raw) if generated_at_raw else None)
    details["generated_at"] = generated_at_raw
    if generated_at is None:
        reasons.append(f"basis_audit_stale:{series}:missing_generated_at")
    else:
        start_of_day = datetime.combine(asof, time(0, 0), tzinfo=ET).astimezone(UTC)
        if generated_at < start_of_day:
            reasons.append(f"basis_audit_stale:{series}:generated_before_day")
        if generated_at > now_utc + timedelta(minutes=5):
            reasons.append(f"basis_audit_stale:{series}:generated_in_future")

    details["sample_count"] = payload.get("sample_count")
    details["basis_quantiles"] = payload.get("basis_quantiles")
    details["flip_risk"] = payload.get("flip_risk")

    flip_payload = payload.get("flip_risk") or {}
    flip_flag = flip_payload.get("flag")
    if isinstance(flip_flag, bool):
        if flip_flag:
            reasons.append(f"basis_flip_risk:{series}")
    else:
        reasons.append(f"basis_flip_risk:{series}:flag_missing")

    details["status"] = "OK" if not reasons else "ALERT"
    return not reasons, reasons, details


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
            reasons.append(f"calibration_missing:{series}:{horizon}:{path.as_posix()}")
            continue
        age = _file_age_days(path, now)
        if age is not None:
            ages[f"{series}:{horizon}"] = age
            if age > max_age_days:
                reasons.append(
                    f"calibration_stale:{series}:{horizon}:{age:.1f}d:{path.as_posix()}"
                )
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
    require_polygon: bool | None = None,
    require_basis_audit: bool | None = None,
    basis_root: Path | None = None,
    series: Sequence[str] | None = None,
    freshness_artifact_path: Path | None = None,
    require_freshness: bool | None = None,
    freshness_scope: str | None = FRESHNESS_SCOPE,
) -> PreflightResult:
    """Evaluate GO/NO-GO checks for index ladder windows."""

    load_env()
    reasons: list[str] = []
    details: dict[str, object] = {}

    reference_et = _ensure_et(now_et)
    now_utc = reference_et.astimezone(UTC)

    # Environment + secrets -------------------------------------------------
    env_missing: list[str] = []
    if require_polygon is None:
        require_polygon = require_kalshi
    if require_freshness is None:
        require_freshness = require_kalshi
    if require_basis_audit is None:
        require_basis_audit = require_kalshi
    if require_kalshi:
        env_missing.extend(_missing_env_vars(["KALSHI_API_KEY_ID", "KALSHI_PRIVATE_KEY_PEM_PATH"]))
        key_path_raw = os.getenv("KALSHI_PRIVATE_KEY_PEM_PATH", "").strip()
        if key_path_raw:
            key_path = Path(key_path_raw).expanduser()
            if not key_path.exists():
                reasons.append("kalshi_private_key_missing")
                details["kalshi_private_key_path"] = key_path.as_posix()
    polygon_key_present = False
    if require_polygon:
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

    # Data freshness --------------------------------------------------------
    if require_freshness:
        from kalshi_alpha.exec.monitors import freshness as freshness_monitor

        enforced_scope = FRESHNESS_SCOPE
        if freshness_scope is not None:
            requested = str(freshness_scope).strip()
            normalized = requested.lower()
            if normalized and normalized not in {"index", "indices", "index-only"}:
                details["freshness_scope_override"] = {
                    "requested": requested,
                    "enforced": enforced_scope,
                }
        freshness_scope = enforced_scope

        artifact_path = (
            Path(freshness_artifact_path)
            if freshness_artifact_path
            else freshness_monitor.FRESHNESS_ARTIFACT_PATH
        )
        payload = freshness_monitor.load_artifact(artifact_path)
        summary = freshness_monitor.summarize_artifact(
            payload,
            artifact_path=artifact_path,
            scope=freshness_scope,
        )
        details["data_freshness"] = summary
        if not summary.get("required_feeds_ok", True):
            reasons.append("STALE_FEEDS")
            stale_feeds = summary.get("stale_feeds") or []
            stale_normalized = {str(feed).strip().lower() for feed in stale_feeds if isinstance(feed, str)}
            if "polygon_index.websocket" in stale_normalized and "polygon_ws_stale" not in reasons:
                reasons.append("polygon_ws_stale")
            if summary.get("status") == "MISSING":
                reasons.append("data_freshness_missing")

    # Polygon connectivity --------------------------------------------------
    if require_polygon and polygon_key_present:
        ping_fn = polygon_ping or _polygon_ping
        if not ping_fn(polygon_timeout):
            reasons.append("polygon_unreachable")

    # Basis audit gate ------------------------------------------------------
    if require_basis_audit:
        asof = reference_et.date()
        series_list = [entry.upper() for entry in (series or _series_labels())]
        root = Path(basis_root) if basis_root else BASIS_AUDIT_ROOT
        audit_details: dict[str, object] = {
            "asof_date": asof.isoformat(),
            "root": root.as_posix(),
            "series": {},
        }
        for series_code in series_list:
            ok, basis_reasons, basis_detail = _check_basis_audit(
                series=series_code,
                asof=asof,
                now_utc=now_utc,
                root=root,
            )
            audit_details["series"][series_code] = basis_detail
            if not ok:
                reasons.extend(basis_reasons)
        details["basis_audit"] = audit_details

    go = not reasons
    details["evaluated_at_et"] = reference_et.isoformat()
    return PreflightResult(go=go, reasons=reasons, details=details)


def _parse_now(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _series_labels(series_horizons: Sequence[tuple[str, str]] = SERIES_HORIZONS) -> tuple[str, ...]:
    return tuple(series for series, _ in series_horizons)


def format_preflight_summary(
    result: PreflightResult,
    *,
    label: str,
    series: Sequence[str] | None = None,
    broker: str | None = None,
) -> str:
    verdict = "GO" if result.go else "NO-GO"
    reasons_count = len(result.reasons)
    series_value = ",".join(series) if series else "ALL"
    line = f"{label}: {verdict} reasons={reasons_count} series={series_value}"
    if broker:
        line = f"{line} (broker={broker})"
    return line


def write_go_no_go_artifact(
    result: PreflightResult,
    *,
    output_path: Path | None = None,
    source: str = "preflight_index",
) -> Path:
    target = Path(output_path) if output_path is not None else GO_NO_GO_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "go": bool(result.go),
        "reasons": list(result.reasons),
        "scope": FRESHNESS_SCOPE,
        "scoped_blockers": list(result.reasons),
        "unscoped_blockers": [],
        "details": dict(result.details),
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "source": source,
    }
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Index ladder preflight checks.")
    parser.add_argument("--now", help="Override current time (ISO-8601, default: now).")
    parser.add_argument("--offline", action="store_true", help="Skip env and Polygon checks.")
    parser.add_argument("--params-root", type=Path, help="Override calibration params root.")
    parser.add_argument("--kill-switch-file", type=Path, help="Override kill switch sentinel path.")
    parser.add_argument(
        "--polygon-timeout",
        type=float,
        default=2.0,
        help="Polygon ping timeout (seconds).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    now = _parse_now(args.now) or datetime.now(tz=UTC)
    result = run_preflight(
        now,
        params_root=args.params_root,
        kill_switch_file=args.kill_switch_file,
        polygon_timeout=float(args.polygon_timeout),
        require_kalshi=not args.offline,
        require_polygon=not args.offline,
    )
    write_go_no_go_artifact(result)
    series = _series_labels()
    print(format_preflight_summary(result, label="PRECHECK index", series=series), flush=True)
    return 0 if result.go else 1


__all__ = [
    "GO_NO_GO_PATH",
    "MAX_CALIBRATION_AGE_DAYS",
    "PreflightResult",
    "format_preflight_summary",
    "main",
    "run_preflight",
    "write_go_no_go_artifact",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
