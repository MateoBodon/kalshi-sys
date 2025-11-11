"""Capital allocator for INX/NDX structures with correlation-aware VaR guardrails."""

from __future__ import annotations

import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Deque, Iterable, Mapping

from kalshi_alpha.risk import CorrelationAwareLimiter, CorrelationConfig
from kalshi_alpha.risk.var_index import SERIES_FAMILY

__all__ = [
    "Allocator",
    "AllocatorConfig",
    "AllocationResult",
    "SeriesBudget",
    "SeriesWindowSample",
    "VarSnapshot",
    "load_scoreboard_history",
    "correlation_var_snapshot",
]

_EPSILON = 1e-9
_DEFAULT_HISTORY_PATH = Path("reports/_artifacts/allocator_history.json")


def _series_key(series: str) -> str:
    return str(series or "").strip().upper()


def _clamp01(value: float | None) -> float:
    if value is None:
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(numeric):
        return 0.0
    return max(0.0, min(1.0, numeric))


@dataclass(frozen=True)
class SeriesWindowSample:
    """Single window observation used to calculate EV×fill×honesty Sharpe."""

    timestamp: datetime
    ev_after_fees: float
    fill_ratio: float
    honesty: float
    weight: float = 1.0

    def signal(self) -> float:
        fill = _clamp01(self.fill_ratio)
        honesty = _clamp01(self.honesty)
        return float(self.ev_after_fees) * fill * honesty

    def normalized_weight(self) -> float:
        return max(float(self.weight), 0.0) or 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp.astimezone(UTC).isoformat(),
            "ev_after_fees": float(self.ev_after_fees),
            "fill_ratio": float(self.fill_ratio),
            "honesty": float(self.honesty),
            "weight": float(self.weight),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> SeriesWindowSample:
        timestamp_raw = payload.get("timestamp")
        if isinstance(timestamp_raw, str):
            try:
                timestamp = datetime.fromisoformat(timestamp_raw)
            except ValueError:
                timestamp = datetime.now(tz=UTC)
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)
        elif isinstance(timestamp_raw, datetime):
            timestamp = timestamp_raw
        else:
            timestamp = datetime.now(tz=UTC)
        return cls(
            timestamp=timestamp,
            ev_after_fees=float(payload.get("ev_after_fees", 0.0)),
            fill_ratio=float(payload.get("fill_ratio", 0.0)),
            honesty=float(payload.get("honesty", 0.0)),
            weight=float(payload.get("weight", 1.0)),
        )


@dataclass(frozen=True)
class RollingStats:
    samples: int
    mean_signal: float
    stdev_signal: float
    sharpe: float


class RollingSharpeWindow:
    """Maintains a rolling EV×fill×honesty Sharpe estimate per series."""

    def __init__(
        self,
        *,
        max_samples: int,
        max_age: timedelta | None,
        vol_floor: float,
    ) -> None:
        self.max_samples = max(1, int(max_samples))
        self.max_age = max_age
        self.vol_floor = max(float(vol_floor), _EPSILON)
        self._samples: Deque[SeriesWindowSample] = deque()

    def extend(self, samples: Iterable[SeriesWindowSample]) -> None:
        for sample in samples:
            self.add(sample)

    def add(self, sample: SeriesWindowSample) -> None:
        self._samples.append(sample)
        self._trim(sample.timestamp)

    def stats(self) -> RollingStats:
        if not self._samples:
            return RollingStats(0, 0.0, self.vol_floor, 0.0)
        values: list[float] = []
        weights: list[float] = []
        for item in self._samples:
            values.append(item.signal())
            weights.append(item.normalized_weight())
        total_weight = sum(weights)
        if total_weight <= 0.0:
            return RollingStats(0, 0.0, self.vol_floor, 0.0)
        mean_signal = sum(value * weight for value, weight in zip(values, weights, strict=True)) / total_weight
        if len(values) == 1:
            stdev = self.vol_floor
        else:
            variance = sum(
                weight * (value - mean_signal) ** 2 for value, weight in zip(values, weights, strict=True)
            ) / max(total_weight - 1.0, 1.0)
            stdev = math.sqrt(max(variance, self.vol_floor**2))
        sharpe = mean_signal / stdev if stdev > _EPSILON else 0.0
        return RollingStats(len(values), mean_signal, stdev, sharpe)

    def snapshot(self) -> list[SeriesWindowSample]:
        return list(self._samples)

    def _trim(self, reference: datetime) -> None:
        while len(self._samples) > self.max_samples:
            self._samples.popleft()
        if self.max_age is None:
            return
        cutoff = reference - self.max_age
        while self._samples and self._samples[0].timestamp < cutoff:
            self._samples.popleft()


@dataclass(frozen=True)
class VarSnapshot:
    """Simplified headroom view derived from the correlation-aware VaR guard."""

    portfolio_limit: float | None = None
    portfolio_var: float | None = None
    family_limits: Mapping[str, float] = field(default_factory=dict)
    family_var: Mapping[str, float] = field(default_factory=dict)

    @property
    def portfolio_headroom(self) -> float:
        if self.portfolio_limit is None:
            return math.inf
        used = max(float(self.portfolio_var or 0.0), 0.0)
        return max(self.portfolio_limit - used, 0.0)

    def headroom(self, series: str) -> float:
        family = SERIES_FAMILY.get(_series_key(series), _series_key(series))
        limit = self.family_limits.get(family)
        if limit is None:
            return math.inf
        used = max(float(self.family_var.get(family, 0.0)), 0.0)
        return max(limit - used, 0.0)

    @classmethod
    def from_guard(
        cls,
        guard: CorrelationAwareLimiter | None,
        *,
        snapshot: Mapping[str, object] | None = None,
    ) -> VarSnapshot:
        if guard is None:
            return cls()
        payload = snapshot or guard.snapshot()
        portfolio_limit = payload.get("portfolio_limit", guard.config.portfolio_limit)
        family_var = payload.get("family_var", {})
        return cls(
            portfolio_limit=float(portfolio_limit) if portfolio_limit is not None else guard.config.portfolio_limit,
            portfolio_var=float(payload.get("portfolio_var", 0.0) or 0.0),
            family_limits=dict(guard.config.family_limits),
            family_var={_series_key(key): float(value) for key, value in dict(family_var).items()},
        )


@dataclass(frozen=True)
class SeriesBudget:
    series: str
    capital: float
    sharpe: float
    signal: float
    volatility: float
    samples: int
    headroom: float
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class AllocationResult:
    series: dict[str, SeriesBudget]
    portfolio_capital: float
    portfolio_headroom: float

    def total_capital(self) -> float:
        return sum(budget.capital for budget in self.series.values())


@dataclass(frozen=True)
class AllocatorConfig:
    window: int = 30
    max_age_days: int = 21
    min_samples: int = 5
    base_capital: float = 5_000.0
    min_sharpe: float = 0.4
    max_series_weight: float = 0.7
    var_buffer: float = 250.0
    vol_floor: float = 25.0
    history_path: Path = _DEFAULT_HISTORY_PATH

    @property
    def max_age(self) -> timedelta:
        return timedelta(days=max(self.max_age_days, 1))


class Allocator:
    """Calculates per-series capital budgets using EV×fill×honesty Sharpe."""

    def __init__(
        self,
        config: AllocatorConfig | None = None,
        *,
        history: Mapping[str, Iterable[SeriesWindowSample]] | None = None,
    ) -> None:
        self.config = config or AllocatorConfig()
        self._windows: dict[str, RollingSharpeWindow] = {}
        if history:
            for series, samples in history.items():
                self.register_many(series, samples)

    def register(self, series: str, sample: SeriesWindowSample) -> None:
        key = _series_key(series)
        window = self._windows.setdefault(
            key,
            RollingSharpeWindow(
                max_samples=self.config.window,
                max_age=self.config.max_age,
                vol_floor=self.config.vol_floor,
            ),
        )
        window.add(sample)

    def register_many(self, series: str, samples: Iterable[SeriesWindowSample]) -> None:
        for sample in samples:
            self.register(series, sample)

    def allocate(self, var_state: VarSnapshot | None = None) -> AllocationResult:
        var_snapshot = var_state or VarSnapshot()
        trackers: dict[str, dict[str, object]] = {}
        for series, window in self._windows.items():
            stats = window.stats()
            trackers[series] = {
                "reasons": [],
                "headroom": var_snapshot.headroom(series),
                "stats": stats,
            }

        eligible: dict[str, RollingStats] = {}
        for series, meta in trackers.items():
            reasons: list[str] = meta["reasons"]
            stats: RollingStats = meta["stats"]
            if stats.samples < self.config.min_samples:
                reasons.append("insufficient_samples")
            if stats.mean_signal <= 0.0:
                reasons.append("negative_signal")
            if stats.sharpe < self.config.min_sharpe:
                reasons.append("low_sharpe")
            if not reasons:
                eligible[series] = stats

        scores: dict[str, float] = {}
        for series, stats in eligible.items():
            scores[series] = max(stats.sharpe * stats.mean_signal, 0.0)

        total_score = sum(scores.values())
        if math.isclose(total_score, 0.0, rel_tol=1e-9, abs_tol=1e-9):
            total_score = 0.0
        portfolio_capital = min(self.config.base_capital, var_snapshot.portfolio_headroom)
        if math.isfinite(portfolio_capital):
            portfolio_capital = max(portfolio_capital - self.config.var_buffer, 0.0)
        else:
            portfolio_capital = self.config.base_capital

        allocations: dict[str, SeriesBudget] = {}
        for series, meta in trackers.items():
            stats: RollingStats = meta["stats"]
            headroom = meta["headroom"]
            reasons = list(meta["reasons"])
            capital = 0.0
            if series in scores and total_score > 0.0 and portfolio_capital > 0.0:
                weight = scores[series] / total_score
                weight = min(weight, self.config.max_series_weight)
                requested = portfolio_capital * weight
                capital = requested
                if headroom < math.inf:
                    allowed = max(headroom - self.config.var_buffer, 0.0)
                    if allowed <= 0.0:
                        capital = 0.0
                        reasons.append("family_headroom")
                    elif allowed + 1e-6 < requested:
                        capital = allowed
                        reasons.append("family_headroom")
            elif series in scores and portfolio_capital <= 0.0:
                reasons.append("portfolio_headroom")

            allocations[series] = SeriesBudget(
                series=series,
                capital=float(round(capital, 2)),
                sharpe=stats.sharpe,
                signal=stats.mean_signal,
                volatility=stats.stdev_signal,
                samples=stats.samples,
                headroom=headroom,
                reasons=tuple(sorted(set(reasons))),
            )

        return AllocationResult(
            series=allocations,
            portfolio_capital=portfolio_capital,
            portfolio_headroom=var_snapshot.portfolio_headroom,
        )

    def dump_history(self) -> dict[str, list[dict[str, object]]]:
        history: dict[str, list[dict[str, object]]] = {}
        for series, window in self._windows.items():
            history[series] = [sample.to_dict() for sample in window.snapshot()]
        return history

    def save_history(self, path: Path | None = None) -> None:
        target = path or self.config.history_path
        target.parent.mkdir(parents=True, exist_ok=True)
        history = self.dump_history()
        target.write_text(json.dumps(history, indent=2), encoding="utf-8")

    @classmethod
    def from_history(cls, path: Path, config: AllocatorConfig | None = None) -> Allocator:
        if not path.exists():
            return cls(config=config)
        payload = json.loads(path.read_text(encoding="utf-8"))
        history: dict[str, list[SeriesWindowSample]] = defaultdict(list)
        if isinstance(payload, Mapping):
            items = payload.items()
        else:
            items = []
        for series, rows in items:
            if not isinstance(rows, list):
                continue
            history[_series_key(series)] = [
                SeriesWindowSample.from_dict(row) for row in rows if isinstance(row, Mapping)
            ]
        return cls(config=config, history=history)


def load_scoreboard_history(path: Path) -> dict[str, list[SeriesWindowSample]]:
    """Load allocator samples from a scoreboard summary JSON payload."""

    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("rows", [])
    history: dict[str, list[SeriesWindowSample]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        series = _series_key(row.get("series", ""))
        if not series:
            continue
        timestamp_raw = row.get("window_end") or row.get("generated_at")
        if isinstance(timestamp_raw, str):
            try:
                timestamp = datetime.fromisoformat(timestamp_raw)
            except ValueError:
                timestamp = datetime.now(tz=UTC)
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)
        else:
            timestamp = datetime.now(tz=UTC)
        sample = SeriesWindowSample(
            timestamp=timestamp,
            ev_after_fees=float(row.get("ev_after_fees", 0.0)),
            fill_ratio=float(row.get("avg_fill_ratio", row.get("avg_fill_ratio_observed", 0.0))),
            honesty=float(row.get("honesty_clamp", 0.0)),
            weight=float(row.get("sample_size", 1)),
        )
        history[series].append(sample)
    return history


def correlation_var_snapshot(config_path: Path | None = None) -> VarSnapshot:
    """Convenience helper returning the baseline VaR snapshot from config only."""

    correlation_config = CorrelationConfig.from_yaml(config_path)
    return VarSnapshot(
        portfolio_limit=correlation_config.portfolio_limit,
        portfolio_var=0.0,
        family_limits=dict(correlation_config.family_limits),
        family_var={},
    )
