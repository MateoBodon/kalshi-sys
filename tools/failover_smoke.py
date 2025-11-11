"""CLI smoke test for DualFeedFailover (synthetic timeline)."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from typing import Iterable, Iterator, Sequence

from kalshi_alpha.data.failover import DualFeedFailover, FeedSample


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate Polygon↔Massive failover using synthetic samples.")
    parser.add_argument("--duration-seconds", type=float, default=3.0, help="Length of simulation horizon.")
    parser.add_argument(
        "--freshness-threshold-ms",
        type=float,
        default=150.0,
        help="Freshness SLO applied to the primary feed.",
    )
    parser.add_argument(
        "--recovery-wait",
        type=float,
        default=0.5,
        help="Seconds to wait before switching back to the primary feed.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Included for backwards compatibility.")
    return parser.parse_args(argv)


def _synthetic_feed(name: str, *, interval_ms: int, outage_start: float | None, outage_duration_ms: int, duration_seconds: float) -> list[FeedSample[int]]:
    base = datetime.now(tz=UTC)
    samples: list[FeedSample[int]] = []
    horizon_ms = int(duration_seconds * 1000)
    t = 0
    outage_triggered = False
    outage_end_ms = None
    if outage_start is not None:
        outage_trigger_ms = int(outage_start * 1000)
        outage_end_ms = outage_trigger_ms + outage_duration_ms
    while t <= horizon_ms:
        if outage_start is not None and not outage_triggered and t >= outage_trigger_ms:
            outage_triggered = True
            t = outage_end_ms if outage_end_ms is not None else t
        samples.append(FeedSample(source=name, timestamp=base + timedelta(milliseconds=t), payload=t))
        t += interval_ms
    return samples


def _merge_samples(primary: Iterable[FeedSample[int]], secondary: Iterable[FeedSample[int]]) -> Iterator[FeedSample[int]]:
    merged = sorted([*primary, *secondary], key=lambda sample: sample.timestamp)
    return iter(merged)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    controller = DualFeedFailover(
        freshness_threshold_ms=args.freshness_threshold_ms,
        recovery_wait_seconds=args.recovery_wait,
    )
    primary = _synthetic_feed(
        "polygon",
        interval_ms=40,
        outage_start=args.duration_seconds / 2,
        outage_duration_ms=400,
        duration_seconds=args.duration_seconds,
    )
    secondary = _synthetic_feed(
        "massive",
        interval_ms=70,
        outage_start=None,
        outage_duration_ms=0,
        duration_seconds=args.duration_seconds,
    )
    active = controller.snapshot()["active_feed"]
    processed = 0
    for sample in _merge_samples(primary, secondary):
        if controller.ingest(sample):
            processed += 1
            if active != controller.snapshot()["active_feed"]:
                active = controller.snapshot()["active_feed"]
                print(f"[failover_smoke] active={active} t={sample.timestamp.isoformat()}")
    print(f"[failover_smoke] processed_samples={processed} final_active={active}")


if __name__ == "__main__":  # pragma: no cover
    main()
