from __future__ import annotations

from datetime import time

import pytest

from kalshi_alpha.strategies.teny import TenYInputs, pmf, pmf_v15


def test_teny_v15_macro_dummies_shift_distribution_center() -> None:
    strikes = [4.3, 4.4, 4.5, 4.6, 4.7]
    history = [4.38, 4.41, 4.43, 4.46, 4.48]
    inputs = TenYInputs(
        prior_close=4.5,
        macro_shock=0.0,
        trailing_history=history,
        macro_shock_dummies={"cpi": 1.0},
    )

    baseline_bins = pmf(strikes, inputs=inputs)
    upgraded_bins = pmf_v15(strikes, inputs=inputs)

    base_central = _central_bin(baseline_bins)
    upgraded_central = _central_bin(upgraded_bins)

    assert upgraded_central.lower > base_central.lower
    assert upgraded_central.upper > base_central.upper


def test_teny_v15_widens_uncertainty_on_event_imbalance() -> None:
    strikes = [4.3, 4.4, 4.5, 4.6, 4.7]
    history = [4.39, 4.42, 4.44, 4.47, 4.49]

    calm_inputs = TenYInputs(
        prior_close=4.5,
        macro_shock=0.0,
        trailing_history=history,
        orderbook_imbalance=0.2,
        event_timestamp=time(14, 55),
    )
    stressed_inputs = TenYInputs(
        prior_close=4.5,
        macro_shock=0.0,
        trailing_history=history,
        orderbook_imbalance=1.1,
        event_timestamp=time(15, 5),
    )

    calm_bins = pmf_v15(strikes, inputs=calm_inputs)
    stressed_bins = pmf_v15(strikes, inputs=stressed_inputs)

    calm_width = _central_bin(calm_bins).upper - _central_bin(calm_bins).lower
    stressed_width = _central_bin(stressed_bins).upper - _central_bin(stressed_bins).lower

    assert stressed_width > calm_width


def test_teny_v15_imbalance_feature_flag_disables_widening() -> None:
    strikes = [4.3, 4.4, 4.5, 4.6, 4.7]
    history = [4.39, 4.42, 4.44, 4.47, 4.49]

    baseline_inputs = TenYInputs(
        prior_close=4.5,
        macro_shock=0.0,
        trailing_history=history,
        orderbook_imbalance=0.2,
        event_timestamp=time(15, 10),
    )
    stressed_enabled = TenYInputs(
        prior_close=4.5,
        macro_shock=0.0,
        trailing_history=history,
        orderbook_imbalance=1.1,
        event_timestamp=time(15, 10),
    )
    stressed_disabled = TenYInputs(
        prior_close=4.5,
        macro_shock=0.0,
        trailing_history=history,
        orderbook_imbalance=1.1,
        event_timestamp=time(15, 10),
        imbalance_feature_enabled=False,
    )

    baseline_bins = pmf_v15(strikes, inputs=baseline_inputs)
    widened_bins = pmf_v15(strikes, inputs=stressed_enabled)
    disabled_bins = pmf_v15(strikes, inputs=stressed_disabled)

    baseline_width = _central_bin(baseline_bins).upper - _central_bin(baseline_bins).lower
    widened_width = _central_bin(widened_bins).upper - _central_bin(widened_bins).lower
    disabled_width = _central_bin(disabled_bins).upper - _central_bin(disabled_bins).lower

    assert widened_width > baseline_width
    assert disabled_width == pytest.approx(baseline_width, rel=1e-6)


def _central_bin(bins):
    for entry in bins:
        if abs(entry.probability - 0.5) < 1e-9:
            return entry
    raise AssertionError("central bin with 0.5 probability not found")
