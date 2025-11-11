"""Service level objective (SLO) aggregations for scoreboard + telemetry exports."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from kalshi_alpha.risk import var_index

LOGGER = logging.getLogger(__name__)
DEFAULT_LOOKBACK_DAYS = 7
SERIES_DEFAULT = ("INX", "INXU", "NASDAQ100", "NASDAQ100U")
SNAPSHOT_STEM_DELIM = "_"


@dataclass(slots=True)
class SLOSeriesMetrics:
    """Container for per-series SLO measurements."""

    series: str
    freshness_p95_ms: float | None = None
    freshness_p99_ms: float | None = None
    time_at_risk_p95_s: float | None = None
    time_at_risk_p99_s: float | None = None
    ev_honesty_brier: float | None = None
    ev_gap_bps: float | None = None
    fill_gap_pp: float | None = None
    var_headroom_usd: float | None = None
    no_go_reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "series": self.series,
            "freshness_p95_ms": self.freshness_p95_ms,
            "freshness_p99_ms": self.freshness_p99_ms,
            "time_at_risk_p95_s": self.time_at_risk_p95_s,
            "time_at_risk_p99_s": self.time_at_risk_p99_s,
            "ev_honesty_brier": self.ev_honesty_brier,
            "ev_gap_bps": self.ev_gap_bps,
            "fill_gap_pp": self.fill_gap_pp,
            "var_headroom_usd": self.var_headroom_usd,
            "no_go_reasons": list(self.no_go_reasons),
        }


def collect_metrics(
    summary: Sequence[Mapping[str, object]] | None,
    *,
    series: Sequence[str] | None = None,
    reports_root: Path = Path("reports"),
    raw_root: Path = Path("data/raw"),
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    now: datetime | None = None,
    var_limits: Mapping[str, float] | None = None,
) -> dict[str, SLOSeriesMetrics]:
    """Build a SLO metrics map keyed by series.

    Parameters
    ----------
    summary:
        Scoreboard summary rows. When ``None`` an empty baseline is created and
        only filesystem-derived metrics (freshness/time-at-risk/VaR headroom)
        are populated.
    series:
        Optional explicit list of series symbols to evaluate. Defaults to
        series present in ``summary`` or the standard index set.
    reports_root / raw_root:
        Roots for persisted markdown reports and raw Polygon snapshots.
    lookback_days:
        Number of trailing days to include when computing latency/headroom
        aggregates.
    now:
        Reference timestamp (UTC). Defaults to ``datetime.now(tz=UTC)``.
    var_limits:
        Optional override for family VaR limits. Defaults to
        ``kalshi_alpha.risk.var_index.load_family_limits``.
    """

    moment = _ensure_utc(now or datetime.now(tz=UTC))
    normalized_limits = {
        key.upper(): float(value)
        for key, value in (var_limits or var_index.load_family_limits()).items()
    }
    base_series: list[str] = []
    metrics: dict[str, SLOSeriesMetrics] = {}

    if summary:
        for row in summary:
            series_label = str(row.get("series") or "").upper()
            if not series_label:
                continue
            base_series.append(series_label)
            metrics[series_label] = SLOSeriesMetrics(
                series=series_label,
                ev_honesty_brier=_safe_float(row.get("honesty_brier")),
                ev_gap_bps=_safe_float(row.get("ev_delta_mean")),
                fill_gap_pp=_fill_gap_percentage(row.get("fill_ratio_vs_alpha")),
                no_go_reasons=[str(reason) for reason in row.get("go_reasons", []) if reason],
            )
    if series:
        base_series.extend(series)
    if not base_series:
        base_series = list(SERIES_DEFAULT)

    for label in base_series:
        label_upper = label.upper()
        metrics.setdefault(label_upper, SLOSeriesMetrics(series=label_upper))

    _populate_freshness(metrics, raw_root=raw_root, lookback_days=lookback_days, now=moment)
    _populate_time_at_risk(metrics, reports_root=reports_root, lookback_days=lookback_days, now=moment)
    _populate_var_headroom(
        metrics,
        reports_root=reports_root,
        lookback_days=lookback_days,
        now=moment,
        var_limits=normalized_limits,
    )
    return metrics


def publish_cloudwatch(
    metrics: Iterable[SLOSeriesMetrics],
    *,
    namespace: str = "KalshiSys/SLO",
    region: str | None = None,
    profile: str | None = None,
) -> None:
    """Push metrics to CloudWatch (best-effort)."""

    payload = list(_cloudwatch_datums(metrics))
    if not payload:
        LOGGER.info("[slo] no SLO metrics to publish")
        return
    try:
        import boto3
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("boto3 is required for CloudWatch publishing") from exc

    session_kwargs: dict[str, object] = {}
    if profile:
        session_kwargs["profile_name"] = profile
    session = boto3.session.Session(**session_kwargs)
    client = session.client("cloudwatch", region_name=region)
    for chunk in _chunk(payload, 20):
        client.put_metric_data(Namespace=namespace, MetricData=chunk)
    LOGGER.info("[slo] published %s CloudWatch datapoints to %s", len(payload), namespace)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute SLO metrics and optionally push to CloudWatch.")
    parser.add_argument("--reports", type=Path, default=Path("reports"), help="Reports root directory.")
    parser.add_argument("--raw", type=Path, default=Path("data/raw"), help="Raw data root (Polygon snapshots).")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK_DAYS, help="Trailing days for filesystem metrics.")
    parser.add_argument(
        "--series",
        nargs="+",
        help="Optional explicit list of series tickers (defaults to summary-derived or standard set).",
    )
    parser.add_argument(
        "--publish-cloudwatch",
        action="store_true",
        help="Publish computed metrics to CloudWatch (requires AWS credentials).",
    )
    parser.add_argument("--cloudwatch-namespace", default="KalshiSys/SLO", help="CloudWatch namespace.")
    parser.add_argument("--cloudwatch-region", default=None, help="CloudWatch region (optional).")
    parser.add_argument("--cloudwatch-profile", default=None, help="Optional AWS profile name.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    metrics = collect_metrics(
        summary=[],
        series=args.series,
        reports_root=args.reports,
        raw_root=args.raw,
        lookback_days=max(args.lookback, 1),
    )
    for item in metrics.values():
        print(json.dumps(item.as_dict(), sort_keys=True))
    if args.publish_cloudwatch:
        publish_cloudwatch(
            metrics.values(),
            namespace=args.cloudwatch_namespace,
            region=args.cloudwatch_region,
            profile=args.cloudwatch_profile,
        )
    return 0


# --- Internal helpers -------------------------------------------------------


def _ensure_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fill_gap_percentage(value: object) -> float | None:
    gap = _safe_float(value)
    if gap is None:
        return None
    return gap * 100.0


def _populate_freshness(
    metrics: Mapping[str, SLOSeriesMetrics],
    *,
    raw_root: Path,
    lookback_days: int,
    now: datetime,
) -> None:
    if lookback_days <= 0:
        return
    for series, slo_metrics in metrics.items():
        latencies = list(
            _iter_snapshot_latencies(
                series=series,
                raw_root=raw_root,
                lookback_days=lookback_days,
                now=now,
            )
        )
        if not latencies:
            continue
        slo_metrics.freshness_p95_ms = _percentile(latencies, 0.95)
        slo_metrics.freshness_p99_ms = _percentile(latencies, 0.99)


def _populate_time_at_risk(
    metrics: Mapping[str, SLOSeriesMetrics],
    *,
    reports_root: Path,
    lookback_days: int,
    now: datetime,
) -> None:
    if lookback_days <= 0:
        return
    threshold = now.date() - timedelta(days=lookback_days)
    for series, slo_metrics in metrics.items():
        samples = list(
            _collect_report_values(
                series=series,
                reports_root=reports_root,
                threshold=threshold,
                pattern="ops_seconds_to_cancel",
            )
        )
        if not samples:
            continue
        slo_metrics.time_at_risk_p95_s = _percentile(samples, 0.95)
        slo_metrics.time_at_risk_p99_s = _percentile(samples, 0.99)


def _populate_var_headroom(
    metrics: Mapping[str, SLOSeriesMetrics],
    *,
    reports_root: Path,
    lookback_days: int,
    now: datetime,
    var_limits: Mapping[str, float],
) -> None:
    if lookback_days <= 0:
        return
    threshold = now.date() - timedelta(days=lookback_days)
    family_limits = var_limits or {}
    for series, slo_metrics in metrics.items():
        family = var_index.SERIES_FAMILY.get(series.upper(), series.upper())
        limit = family_limits.get(family)
        if limit is None:
            continue
        samples = list(
            _collect_report_values(
                series=series,
                reports_root=reports_root,
                threshold=threshold,
                pattern="Portfolio VaR",
            )
        )
        if not samples:
            continue
        max_var = max(samples)
        slo_metrics.var_headroom_usd = max(limit - max_var, 0.0)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    percentile = min(max(percentile, 0.0), 1.0)
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = int(round(percentile * (len(ordered) - 1)))
    return ordered[index]


def _iter_snapshot_latencies(
    *,
    series: str,
    raw_root: Path,
    lookback_days: int,
    now: datetime,
    max_samples: int = 5000,
) -> Iterable[float]:
    ticker = series.upper()
    collected = 0
    for day_offset in range(lookback_days):
        day = (now - timedelta(days=day_offset)).date()
        day_dir = raw_root / f"{day.year:04d}" / f"{day.month:02d}" / f"{day.day:02d}" / "polygon_index"
        if not day_dir.exists():
            continue
        for path in sorted(day_dir.glob(f"*_{ticker}_snapshot*.json*"), reverse=True):
            latency = _snapshot_latency_ms(path)
            if latency is None:
                continue
            yield latency
            collected += 1
            if collected >= max_samples:
                return


def _snapshot_latency_ms(path: Path) -> float | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    timestamp_raw = payload.get("timestamp")
    if not isinstance(timestamp_raw, str):
        return None
    try:
        snapshot_ts = datetime.fromisoformat(timestamp_raw)
    except ValueError:
        return None
    if snapshot_ts.tzinfo is None:
        snapshot_ts = snapshot_ts.replace(tzinfo=UTC)
    else:
        snapshot_ts = snapshot_ts.astimezone(UTC)
    stem = path.name.split(SNAPSHOT_STEM_DELIM, 1)[0]
    try:
        ingest = datetime.strptime(stem, "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
    except ValueError:
        ingest = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    latency_ms = (ingest - snapshot_ts).total_seconds() * 1000.0
    return max(latency_ms, 0.0)


def _collect_report_values(
    *,
    series: str,
    reports_root: Path,
    threshold: date,
    pattern: str,
) -> Iterable[float]:
    for path in _series_report_paths(series=series, reports_root=reports_root):
        report_date = _parse_report_date(path)
        if report_date is None or report_date < threshold:
            continue
        try:
            contents = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for value in _extract_numeric(contents, pattern):
            yield value


def _series_report_paths(*, series: str, reports_root: Path) -> Iterable[Path]:
    series_dir = reports_root / series.upper()
    if series_dir.exists():
        yield from series_dir.glob("*.md")
    ladders_dir = reports_root / "index_ladders" / series.upper()
    if ladders_dir.exists():
        yield from ladders_dir.glob("*.md")
    hourly_root = reports_root / "index_hourly"
    if hourly_root.exists():
        for mode_dir in hourly_root.iterdir():
            candidate = mode_dir / series.upper()
            if candidate.exists():
                yield from candidate.glob("*.md")


def _parse_report_date(path: Path) -> date | None:
    try:
        return date.fromisoformat(path.stem)
    except ValueError:
        return None


def _extract_numeric(contents: str, pattern: str) -> Iterable[float]:
    needle = f"{pattern}:"
    for line in contents.splitlines():
        if needle not in line:
            continue
        try:
            number_text = line.split(needle, 1)[1].strip()
            token = number_text.split()[0]
            yield float(token)
        except (IndexError, ValueError):
            continue


def _cloudwatch_datums(metrics: Iterable[SLOSeriesMetrics]) -> Iterable[dict[str, object]]:
    timestamp = datetime.now(tz=UTC)
    for slo_metrics in metrics:
        dims = [{"Name": "Series", "Value": slo_metrics.series}]
        for name, value, unit in (
            ("FreshnessP95", slo_metrics.freshness_p95_ms, "Milliseconds"),
            ("FreshnessP99", slo_metrics.freshness_p99_ms, "Milliseconds"),
            ("TimeAtRiskP95", slo_metrics.time_at_risk_p95_s, "Seconds"),
            ("TimeAtRiskP99", slo_metrics.time_at_risk_p99_s, "Seconds"),
            ("EVGapBps", slo_metrics.ev_gap_bps, "None"),
            ("FillGapPp", slo_metrics.fill_gap_pp, "None"),
            ("HonestyBrier", slo_metrics.ev_honesty_brier, "None"),
            ("VaRHeadroom", slo_metrics.var_headroom_usd, "None"),
        ):
            if value is None:
                continue
            yield {
                "MetricName": name,
                "Dimensions": dims,
                "Timestamp": timestamp,
                "Value": float(value),
                "Unit": unit,
            }
        if slo_metrics.no_go_reasons:
            yield {
                "MetricName": "NoGoActive",
                "Dimensions": dims,
                "Timestamp": timestamp,
                "Value": 1.0,
                "Unit": "Count",
            }


def _chunk(sequence: Sequence[dict[str, object]] | list[dict[str, object]], size: int) -> Iterable[list[dict[str, object]]]:
    for start in range(0, len(sequence), size):
        yield sequence[start : start + size]


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
