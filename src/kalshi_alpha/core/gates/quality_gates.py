"""Production quality gates for daily orchestration."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl
import yaml

from kalshi_alpha.datastore.paths import PROC_ROOT, RAW_ROOT

ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class MetricThreshold:
    series: str
    crps_advantage_min: float | None = None
    brier_advantage_min: float | None = None


@dataclass(frozen=True)
class DataFreshnessThreshold:
    name: str
    namespace: str
    timestamp_field: str = "as_of"
    max_age: timedelta = timedelta(hours=24)
    require_et: bool = True


@dataclass(frozen=True)
class ReconciliationThreshold:
    name: str
    namespace: str
    par_maturity: str
    dgs_maturity: str
    tolerance_bps: float


@dataclass(frozen=True)
class QualityGateConfig:
    metrics: list[MetricThreshold]
    data_freshness: list[DataFreshnessThreshold]
    reconciliations: list[ReconciliationThreshold]
    monitor_limits: dict[str, float]


@dataclass(frozen=True)
class QualityGateResult:
    go: bool
    reasons: list[str]
    details: dict[str, Any]


def load_quality_gate_config(path: Path | None = None) -> QualityGateConfig:
    """Load quality gate thresholds from YAML."""

    source_path: Path | None
    if path is not None:
        source_path = Path(path)
    else:
        candidate = Path("configs/quality_gates.yaml")
        if candidate.exists():
            source_path = candidate
        else:
            example = Path("configs/quality_gates.example.yaml")
            source_path = example if example.exists() else None

    if source_path is None or not source_path.exists():
        raise FileNotFoundError("quality gate config not found")

    payload = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}

    metrics_section = payload.get("metrics", {})
    default_metrics = metrics_section.get("default", {}) if isinstance(metrics_section, dict) else {}
    series_cfg: dict[str, Any]
    if isinstance(metrics_section, dict):
        series_cfg = metrics_section.get("series", {}) or {}
    else:
        series_cfg = {}
    metrics: list[MetricThreshold] = []
    for raw_series, cfg in series_cfg.items():
        merged = {**default_metrics, **(cfg or {})}
        metrics.append(
            MetricThreshold(
                series=str(raw_series).lower(),
                crps_advantage_min=_maybe_float(merged.get("crps_advantage_min")),
                brier_advantage_min=_maybe_float(merged.get("brier_advantage_min")),
            )
        )

    raw_sources = payload.get("data_freshness", {})
    if isinstance(raw_sources, dict):
        sources_iterable: Iterable[dict[str, Any]] = raw_sources.get("sources", []) or []
    elif isinstance(raw_sources, list):
        sources_iterable = raw_sources
    else:
        sources_iterable = []

    data_freshness = []
    for item in sources_iterable:
        if not isinstance(item, dict):
            continue
        max_age_seconds = _maybe_float(item.get("max_age_seconds"))
        max_age_minutes = _maybe_float(item.get("max_age_minutes"))
        max_age_hours = _maybe_float(item.get("max_age_hours"), default=24.0)
        if max_age_seconds is not None:
            max_age = timedelta(seconds=max_age_seconds)
        elif max_age_minutes is not None:
            max_age = timedelta(minutes=max_age_minutes)
        else:
            max_age = timedelta(hours=max_age_hours if max_age_hours is not None else 24.0)
        data_freshness.append(
            DataFreshnessThreshold(
                name=str(item.get("name", "unknown")),
                namespace=str(item.get("namespace", "")).strip(),
                timestamp_field=str(item.get("timestamp_field", "as_of")),
                max_age=max_age,
                require_et=bool(item.get("require_et", True)),
            )
        )

    raw_recon = payload.get("reconciliation", {})
    if isinstance(raw_recon, dict):
        recon_iterable: Iterable[dict[str, Any]] = raw_recon.get("checks", []) or []
        # Backwards compat: allow single mapping at top-level
        if not recon_iterable and {"name", "namespace", "par_maturity", "dgs_maturity"}.issubset(raw_recon):
            recon_iterable = [raw_recon]
    elif isinstance(raw_recon, list):
        recon_iterable = raw_recon
    else:
        recon_iterable = []

    reconciliations = []
    for item in recon_iterable:
        if not isinstance(item, dict):
            continue
        tolerance = _maybe_float(item.get("tolerance_bps"), default=5.0)
        reconciliations.append(
            ReconciliationThreshold(
                name=str(item.get("name", "treasury")),
                namespace=str(item.get("namespace", "")).strip(),
                par_maturity=str(item.get("par_maturity", "10 YR")),
                dgs_maturity=str(item.get("dgs_maturity", "DGS10")),
                tolerance_bps=tolerance if tolerance is not None else 5.0,
            )
        )

    monitors = payload.get("monitors", {})
    monitor_limits = {}
    if isinstance(monitors, dict):
        for key, value in monitors.items():
            numeric = _maybe_float(value, default=0.0)
            if numeric is not None:
                monitor_limits[str(key)] = numeric

    return QualityGateConfig(
        metrics=metrics,
        data_freshness=data_freshness,
        reconciliations=reconciliations,
        monitor_limits=monitor_limits,
    )


def run_quality_gates(
    *,
    config: QualityGateConfig | None = None,
    config_path: Path | None = None,
    monitors: dict[str, Any] | None = None,
    now: datetime | None = None,
    proc_root: Path | None = None,
    raw_root: Path | None = None,
) -> QualityGateResult:
    """Evaluate all gates and return go/no-go verdict."""

    cfg = config or load_quality_gate_config(config_path)
    evaluator = _QualityGateEvaluator(
        cfg,
        monitors or {},
        now or datetime.now(tz=UTC),
        proc_root or PROC_ROOT,
        raw_root or RAW_ROOT,
    )
    return evaluator.evaluate()


def _maybe_float(value: object, *, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


class _QualityGateEvaluator:
    def __init__(
        self,
        config: QualityGateConfig,
        monitors: dict[str, Any],
        now: datetime,
        proc_root: Path,
        raw_root: Path,
    ) -> None:
        self._config = config
        self._monitors = monitors
        self._now_utc = now.astimezone(UTC)
        self._now_et = self._now_utc.astimezone(ET)
        self._proc_root = proc_root
        self._raw_root = raw_root
        self._reasons: list[str] = []
        self._details: dict[str, Any] = {}

    def evaluate(self) -> QualityGateResult:
        self._check_metrics()
        self._check_data_freshness()
        self._check_reconciliations()
        self._check_monitors()
        return QualityGateResult(go=not self._reasons, reasons=self._reasons, details=self._details)

    # --- individual checks -------------------------------------------------

    def _check_metrics(self) -> None:
        for threshold in self._config.metrics:
            series_slug = threshold.series.lower()
            calib_path = self._proc_root / f"{series_slug}_calib.parquet"
            key_prefix = f"metrics.{series_slug}"
            if not calib_path.exists():
                self._reasons.append(f"{key_prefix}.missing_calibration")
                continue
            try:
                frame = pl.read_parquet(calib_path)
            except Exception as exc:  # pragma: no cover - IO guard
                self._reasons.append(f"{key_prefix}.unreadable:{exc}")
                continue
            params = frame.filter(pl.col("record_type") == "params")
            if params.is_empty():
                self._reasons.append(f"{key_prefix}.missing_params")
                continue
            row = params.row(0, named=True)

            if threshold.crps_advantage_min is not None:
                crps = _maybe_float(row.get("crps"))
                baseline = _maybe_float(row.get("baseline_crps"))
                if crps is None or baseline is None or math.isnan(crps) or math.isnan(baseline):
                    self._reasons.append(f"{key_prefix}.crps_missing")
                else:
                    advantage = baseline - crps
                    self._details[f"{key_prefix}.crps_advantage"] = advantage
                    if advantage < threshold.crps_advantage_min:
                        self._reasons.append(
                            f"{key_prefix}.crps_advantage:{advantage:.4f}<{threshold.crps_advantage_min:.4f}"
                        )

            if threshold.brier_advantage_min is not None:
                brier = _maybe_float(row.get("brier"))
                baseline_brier = _maybe_float(row.get("baseline_brier"))
                if brier is None or baseline_brier is None or math.isnan(brier) or math.isnan(baseline_brier):
                    self._reasons.append(f"{key_prefix}.brier_missing")
                else:
                    advantage = baseline_brier - brier
                    self._details[f"{key_prefix}.brier_advantage"] = advantage
                    if advantage < threshold.brier_advantage_min:
                        self._reasons.append(
                            f"{key_prefix}.brier_advantage:{advantage:.4f}<{threshold.brier_advantage_min:.4f}"
                        )

    def _check_data_freshness(self) -> None:
        for item in self._config.data_freshness:
            key_prefix = f"data_freshness.{item.name}"
            namespace_dir = self._proc_root / item.namespace
            if not namespace_dir.exists():
                self._reasons.append(f"{key_prefix}.missing_namespace")
                continue
            latest = _latest_parquet(namespace_dir)
            if latest is None:
                self._reasons.append(f"{key_prefix}.missing_snapshot")
                continue
            try:
                frame = pl.read_parquet(latest)
            except Exception as exc:  # pragma: no cover - IO guard
                self._reasons.append(f"{key_prefix}.unreadable:{exc}")
                continue
            if item.timestamp_field not in frame.columns:
                self._reasons.append(f"{key_prefix}.missing_field:{item.timestamp_field}")
                continue
            value = frame.select(pl.col(item.timestamp_field).max()).item()
            if value is None:
                self._reasons.append(f"{key_prefix}.no_timestamp")
                continue
            if isinstance(value, datetime):
                as_of = value
            else:
                # accept polars date objects
                close_time = time(16, 0)
                if item.require_et:
                    as_of = datetime.combine(value, close_time, tzinfo=ET).astimezone(UTC)
                else:
                    as_of = datetime.combine(value, close_time, tzinfo=UTC)
            if as_of.tzinfo is None:
                self._reasons.append(f"{key_prefix}.timezone_missing")
                as_of = as_of.replace(tzinfo=ET if item.require_et else UTC)
            as_of_et = as_of.astimezone(ET)
            age = self._now_et - as_of_et
            if age.total_seconds() < 0:
                age = timedelta(seconds=0)
            self._details[f"{key_prefix}.age_hours"] = round(age.total_seconds() / 3600, 3)
            if age > item.max_age:
                self._reasons.append(
                    f"{key_prefix}.stale:{age.total_seconds()/3600:.2f}h>{item.max_age.total_seconds()/3600:.2f}h"
                )

    def _check_reconciliations(self) -> None:
        for item in self._config.reconciliations:
            key_prefix = f"reconciliation.{item.name}"
            namespace_dir = self._proc_root / item.namespace
            if not namespace_dir.exists():
                self._reasons.append(f"{key_prefix}.missing_namespace")
                continue
            latest = _latest_parquet(namespace_dir)
            if latest is None:
                self._reasons.append(f"{key_prefix}.missing_snapshot")
                continue
            try:
                frame = pl.read_parquet(latest)
            except Exception as exc:  # pragma: no cover - IO guard
                self._reasons.append(f"{key_prefix}.unreadable:{exc}")
                continue
            required_cols = {"as_of", "maturity", "rate"}
            if not required_cols.issubset(frame.columns):
                self._reasons.append(f"{key_prefix}.missing_columns")
                continue
            diffs: list[tuple[Any, float]] = []
            grouped = (
                frame.with_columns(pl.col("maturity").str.to_uppercase())
                .group_by("as_of")
                .agg([pl.col("maturity"), pl.col("rate")])
            )
            overlap_found = False
            for row in grouped.iter_rows(named=True):
                maturities = row["maturity"]
                rates = row["rate"]
                pairs = dict(zip(maturities, rates, strict=False))
                par_rate = pairs.get(item.par_maturity.upper())
                dgs_rate = pairs.get(item.dgs_maturity.upper())
                if par_rate is None or dgs_rate is None:
                    continue
                overlap_found = True
                diff_bps = abs(float(par_rate) - float(dgs_rate)) * 100.0
                diffs.append((row["as_of"], diff_bps))
                if diff_bps > item.tolerance_bps:
                    self._reasons.append(
                        f"{key_prefix}.diff:{diff_bps:.2f}bps>{item.tolerance_bps:.2f}bps@{row['as_of']}"
                        f"[{item.par_maturity.upper()}-{item.dgs_maturity.upper()}]@{latest.name}"
                    )
            if not overlap_found:
                self._reasons.append(
                    f"{key_prefix}.no_overlap"
                    f"[{item.par_maturity.upper()}-{item.dgs_maturity.upper()}]@{latest.name}"
                )
            if diffs:
                worst = max(diffs, key=lambda pair: pair[1])
                self._details[f"{key_prefix}.max_diff_bps"] = round(worst[1], 3)

    def _check_monitors(self) -> None:
        for key, limit in self._config.monitor_limits.items():
            value = self._monitors.get(key, 0)
            if isinstance(value, bool):
                numeric = 1.0 if value else 0.0
            elif isinstance(value, (int, float)) and not math.isnan(float(value)):
                numeric = float(value)
            else:
                numeric = 0.0
            self._details[f"monitors.{key}"] = numeric
            if numeric > limit:
                self._reasons.append(
                    f"monitors.{key}:{numeric:.2f}>{float(limit):.2f}"
                )


def _latest_parquet(directory: Path) -> Path | None:
    candidates = sorted(directory.glob("*.parquet"))
    if not candidates:
        return None
    return candidates[-1]
