from __future__ import annotations

from datetime import UTC, datetime

from kalshi_alpha.strategies.teny import TenYInputs, pmf_v15


def test_teny_imbalance_widens_spread() -> None:
    strikes = [4.10, 4.20, 4.30]
    history = [4.00, 4.05, 4.15, 4.20]
    event_time = datetime(2025, 1, 6, 20, 10, tzinfo=UTC)  # 15:10 ET

    baseline_inputs = TenYInputs(
        prior_close=4.20,
        macro_shock=0.0,
        trailing_history=history,
        event_timestamp=event_time,
    )
    balanced = pmf_v15(strikes, inputs=baseline_inputs)

    widened_inputs = TenYInputs(
        prior_close=4.20,
        macro_shock=0.0,
        trailing_history=history,
        orderbook_imbalance=0.85,
        event_timestamp=event_time,
    )
    widened = pmf_v15(strikes, inputs=widened_inputs)

    assert balanced[0].upper is not None
    assert balanced[-1].lower is not None
    assert widened[0].upper is not None
    assert widened[-1].lower is not None

    wide_span = widened[-1].lower - widened[0].upper
    base_span = balanced[-1].lower - balanced[0].upper
    assert wide_span > base_span
