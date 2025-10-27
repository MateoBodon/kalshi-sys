"""Kelly-based sizing helpers with safety caps."""

from __future__ import annotations


def kelly_yes_no(p_true: float, p_mkt: float) -> float:
    """Return the raw Kelly fraction for taking the YES side."""
    if not 0.0 <= p_true <= 1.0:
        raise ValueError("p_true must be within [0, 1]")
    if not 0.0 < p_mkt < 1.0:
        raise ValueError("p_mkt must be within (0, 1)")
    edge = p_true - p_mkt
    denom = 1.0 - p_mkt
    if denom <= 0.0:
        raise ValueError("Market price must be strictly less than 1.0")
    return edge / denom


def truncate_kelly(kelly_fraction: float, cap: float) -> float:
    """Truncate a Kelly fraction to a symmetric +/- cap."""
    if cap <= 0.0:
        raise ValueError("cap must be positive")
    return max(min(kelly_fraction, cap), -cap)


def scale_kelly(kelly_fraction: float, uncertainty: float, ob_imbalance: float, cap: float) -> float:
    """Scale a Kelly fraction by uncertainty and orderbook imbalance penalties."""

    if cap <= 0.0:
        raise ValueError("cap must be positive")

    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))

    baseline = truncate_kelly(kelly_fraction, cap)
    if baseline == 0.0:
        return 0.0

    penalty_uncertainty = 1.0 - _clamp(uncertainty)
    penalty_imbalance = 1.0 - _clamp(ob_imbalance)
    scale = max(0.0, penalty_uncertainty) * max(0.0, penalty_imbalance)
    scaled = baseline * scale
    return truncate_kelly(scaled, cap)


def apply_caps(
    size: float,
    pal: float | None,
    max_loss_per_strike: float | None,
    max_var: float | None,
) -> float:
    """Apply PAL, per-strike, and VaR caps to a proposed risk size."""
    if size <= 0.0:
        return 0.0
    capped = size
    for cap in (pal, max_loss_per_strike, max_var):
        if cap is None:
            continue
        if cap <= 0.0:
            return 0.0
        capped = min(capped, cap)
    return max(capped, 0.0)
