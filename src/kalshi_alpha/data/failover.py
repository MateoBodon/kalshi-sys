"""SLO-enforced dual-feed failover controller."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class FeedSample(Generic[T]):
    """Normalized feed sample."""

    source: str
    timestamp: datetime
    payload: T


class DualFeedFailover(Generic[T]):
    """Track two feeds (Polygon primary, Massive secondary) and decide which to trust."""

    def __init__(
        self,
        *,
        primary_name: str = "polygon",
        secondary_name: str = "massive",
        freshness_threshold_ms: float = 1200.0,
        recovery_wait_seconds: float = 5.0,
    ) -> None:
        self.primary_name = primary_name
        self.secondary_name = secondary_name
        self.freshness_threshold_ms = max(float(freshness_threshold_ms), 1.0)
        self.recovery_wait_seconds = max(float(recovery_wait_seconds), 0.0)
        self._active_feed = self.primary_name
        self._last_sample: dict[str, datetime] = {}
        self._last_switch: datetime | None = None

    def ingest(self, sample: FeedSample[T]) -> bool:
        """Record a sample and return True if it should be emitted downstream."""

        normalized_ts = _ensure_utc(sample.timestamp)
        self._last_sample[sample.source] = normalized_ts
        self._evaluate_active(normalized_ts)
        return sample.source == self._active_feed

    def snapshot(self) -> dict[str, object]:
        return {
            "active_feed": self._active_feed,
            "primary_age_ms": self._age_ms(self.primary_name, datetime.now(tz=UTC)),
            "secondary_age_ms": self._age_ms(self.secondary_name, datetime.now(tz=UTC)),
            "last_switch_at": self._last_switch.isoformat() if self._last_switch else None,
        }

    def _evaluate_active(self, reference: datetime) -> None:
        now = _ensure_utc(reference)
        primary_age = self._age_ms(self.primary_name, now)
        secondary_age = self._age_ms(self.secondary_name, now)
        if self._active_feed == self.primary_name:
            if primary_age is None:
                return
            if primary_age > self.freshness_threshold_ms and secondary_age is not None and secondary_age <= self.freshness_threshold_ms:
                self._active_feed = self.secondary_name
                self._last_switch = now
        else:
            if secondary_age is not None and secondary_age > self.freshness_threshold_ms:
                return
            if (
                primary_age is not None
                and primary_age <= self.freshness_threshold_ms
                and self._ready_for_recovery(now)
            ):
                self._active_feed = self.primary_name
                self._last_switch = now

    def _ready_for_recovery(self, now: datetime) -> bool:
        if self._last_switch is None:
            return True
        return (now - self._last_switch).total_seconds() >= self.recovery_wait_seconds

    def _age_ms(self, feed: str, now: datetime) -> float | None:
        timestamp = self._last_sample.get(feed)
        if timestamp is None:
            return None
        return max((now - timestamp).total_seconds() * 1000.0, 0.0)


def _ensure_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


__all__ = ["DualFeedFailover", "FeedSample"]
