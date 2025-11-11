from __future__ import annotations

from datetime import UTC, datetime, timedelta

from kalshi_alpha.data.failover import DualFeedFailover, FeedSample


def _sample(source: str, offset_ms: int) -> FeedSample[int]:
    base = datetime.now(tz=UTC)
    return FeedSample(source=source, timestamp=base + timedelta(milliseconds=offset_ms), payload=offset_ms)


def test_failover_switches_to_secondary() -> None:
    controller = DualFeedFailover(freshness_threshold_ms=50.0)
    assert controller.ingest(_sample("polygon", 0))
    secondary_sample = _sample("massive", 120)
    assert controller.ingest(secondary_sample)
    snapshot = controller.snapshot()
    assert snapshot["active_feed"] == "massive"


def test_recovery_back_to_primary() -> None:
    controller = DualFeedFailover(freshness_threshold_ms=50.0, recovery_wait_seconds=0.0)
    controller.ingest(_sample("polygon", 0))
    controller.ingest(_sample("massive", 120))
    recovery_sample = _sample("polygon", 140)
    assert controller.ingest(recovery_sample)
    assert controller.snapshot()["active_feed"] == "polygon"
