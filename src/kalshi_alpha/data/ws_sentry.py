"""Lightweight websocket freshness sentry with strict (final-minute) gating."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


@dataclass
class WSFreshnessSentry:
    """Track websocket update latency and expose strict freshness checks."""

    soft_threshold_ms: float = 1500.0
    strict_threshold_ms: float = 700.0

    def __post_init__(self) -> None:
        self._last_sample_utc: datetime | None = None
        self._last_latency_ms: float | None = None

    def record_timestamp(self, sample_timestamp: datetime, *, now: datetime | None = None) -> float:
        """Record an update emitted at ``sample_timestamp`` and return latency in ms."""

        reference = _ensure_utc(now or datetime.now(tz=UTC))
        sample_utc = _ensure_utc(sample_timestamp)
        latency_ms = max((reference - sample_utc).total_seconds() * 1000.0, 0.0)
        self._last_sample_utc = sample_utc
        self._last_latency_ms = latency_ms
        return latency_ms

    def record_latency(self, latency_ms: float, *, now: datetime | None = None) -> float:
        """Record a pre-computed latency measurement (milliseconds)."""

        reference = _ensure_utc(now or datetime.now(tz=UTC))
        value = max(float(latency_ms), 0.0)
        self._last_sample_utc = reference - timedelta(milliseconds=value)
        self._last_latency_ms = value
        return value

    def age_ms(self, now: datetime | None = None) -> float | None:
        """Return the current staleness in milliseconds."""

        if self._last_sample_utc is None:
            return self._last_latency_ms
        reference = _ensure_utc(now or datetime.now(tz=UTC))
        return max((reference - self._last_sample_utc).total_seconds() * 1000.0, 0.0)

    def is_fresh(self, *, strict: bool, now: datetime | None = None) -> bool:
        """Check whether the latest update meets freshness thresholds."""

        current_age = self.age_ms(now)
        if current_age is None:
            return False
        threshold = self.strict_threshold_ms if strict else self.soft_threshold_ms
        return current_age <= threshold


def _ensure_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


__all__ = ["WSFreshnessSentry"]
