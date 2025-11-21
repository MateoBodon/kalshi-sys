"""Lightweight maker fill probability heuristic for index ladders."""

from __future__ import annotations

import math

FILL_MIN = 0.02
FILL_MAX = 0.98


def estimate_maker_fill_prob(
    distance_to_mid_cents: float,
    time_to_expiry_minutes: float,
    spread_cents: float,
) -> float:
    """Return a conservative fill probability for a maker quote.

    Heuristics:
    - Close to mid (or inside the spread) => higher chance.
    - More time to expiry => higher chance.
    - Very wide spreads reduce the penalty for being off-mid.
    """

    dist = max(distance_to_mid_cents, 0.0)
    spread = max(spread_cents, 1e-6)
    time_left = max(time_to_expiry_minutes, 0.0)

    # Distance factor: logistic drop-off as you move away from mid, scaled by spread.
    normalized_dist = dist / max(spread, 1.0)
    distance_factor = 1.0 / (1.0 + math.exp(1.6 * (normalized_dist - 0.5)))

    # Time factor: quick rise early, then taper.
    time_factor = 1.0 - math.exp(-time_left / 45.0)

    # Small bonus for being at/inside the spread.
    inside_bonus = 0.05 if dist <= 0.25 * spread else 0.0

    prob = (distance_factor * time_factor) + inside_bonus
    return max(FILL_MIN, min(FILL_MAX, prob))


__all__ = ["estimate_maker_fill_prob"]
