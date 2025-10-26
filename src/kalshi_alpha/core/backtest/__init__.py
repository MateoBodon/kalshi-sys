"""Backtesting utilities: scoring rules, event replay, and calibration diagnostics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from math import isfinite, log

from kalshi_alpha.core.pricing import LadderBinProbability

EPS = 1e-12


def brier_score(probabilities: Sequence[float], outcomes: Sequence[int]) -> float:
    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must have the same length")
    total = 0.0
    for prob, outcome in zip(probabilities, outcomes, strict=True):
        if outcome not in (0, 1):
            raise ValueError("outcomes must be 0 or 1 for Brier score")
        total += (prob - outcome) ** 2
    return total / len(probabilities) if probabilities else 0.0


def log_loss(
    probabilities: Sequence[float],
    outcomes: Sequence[int],
    *,
    epsilon: float = EPS,
) -> float:
    if len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must have the same length")
    total = 0.0
    for prob, outcome in zip(probabilities, outcomes, strict=True):
        if outcome not in (0, 1):
            raise ValueError("outcomes must be 0 or 1 for log-loss")
        clipped = min(max(prob, epsilon), 1.0 - epsilon)
        total += outcome * -log(clipped) + (1 - outcome) * -log(1 - clipped)
    return total / len(probabilities) if probabilities else 0.0


def crps_from_pmf(pmf: Sequence[LadderBinProbability], observation: float) -> float:
    """Continuous Ranked Probability Score for discrete ladder PMFs."""
    cumulative = 0.0
    crps = 0.0
    for bin_prob in pmf:
        cumulative += bin_prob.probability
        indicator = 1.0 if _bin_contains(bin_prob, observation) else 0.0
        crps += (cumulative - indicator) ** 2 * _bin_width(bin_prob)
    return crps


def _bin_contains(bin_prob: LadderBinProbability, observation: float) -> bool:
    lower = float("-inf") if bin_prob.lower is None else bin_prob.lower
    upper = float("inf") if bin_prob.upper is None else bin_prob.upper
    return lower <= observation < upper


def _bin_width(bin_prob: LadderBinProbability) -> float:
    lower = float("-inf") if bin_prob.lower is None else bin_prob.lower
    upper = float("inf") if bin_prob.upper is None else bin_prob.upper
    if not (isfinite(lower) and isfinite(upper)):
        return 1.0  # fallback for open-ended tails
    return max(upper - lower, 1e-6)


@dataclass(frozen=True)
class Snapshot:
    timestamp: datetime
    payload: dict


class EventReplayer:
    """Simple chronological iterator over event snapshots."""

    def __init__(self, snapshots: Iterable[Snapshot]) -> None:
        self._snapshots = sorted(snapshots, key=lambda snap: snap.timestamp)

    def iter_until(self, cutoff: datetime) -> Iterable[Snapshot]:
        for snapshot in self._snapshots:
            if snapshot.timestamp <= cutoff:
                yield snapshot

    def latest_before(self, cutoff: datetime) -> Snapshot | None:
        candidate = None
        for snapshot in self._snapshots:
            if snapshot.timestamp <= cutoff:
                candidate = snapshot
            else:
                break
        return candidate


def reliability_table(
    probabilities: Sequence[float],
    outcomes: Sequence[int],
    *,
    buckets: int = 10,
) -> list[dict]:
    """Bucketed calibration table (reliability curve)."""
    if buckets <= 0:
        raise ValueError("buckets must be positive")
    bucket_data: list[list[tuple[float, int]]] = [[] for _ in range(buckets)]
    for prob, outcome in zip(probabilities, outcomes, strict=True):
        index = min(int(prob * buckets), buckets - 1)
        bucket_data[index].append((prob, outcome))

    table: list[dict] = []
    for idx, entries in enumerate(bucket_data):
        if not entries:
            avg_prob = 0.0
            event_rate = 0.0
            count = 0
        else:
            probs, outs = zip(*entries, strict=True)
            avg_prob = sum(probs) / len(probs)
            event_rate = sum(outs) / len(outs)
            count = len(entries)
        table.append(
            {
                "bucket": idx,
                "avg_probability": avg_prob,
                "event_rate": event_rate,
                "count": count,
            }
        )
    return table


def probability_integral_transform(
    pmf: Sequence[LadderBinProbability],
    observation: float,
) -> float:
    """Discrete PIT using ladder PMFs."""
    cumulative = 0.0
    for bin_prob in pmf:
        if bin_prob.upper is None or observation < bin_prob.upper:
            cumulative += bin_prob.probability
            return cumulative
        cumulative += bin_prob.probability
    return 1.0
