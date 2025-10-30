from __future__ import annotations

import pytest

from kalshi_alpha.core.execution.fillratio import FillRatioEstimator, alpha_row
from kalshi_alpha.core.kalshi_api import Orderbook


def _orderbook(side_price: float, depth: float) -> Orderbook:
    return Orderbook(
        market_id="MKT",
        bids=[{"price": side_price, "size": depth}],
        asks=[{"price": side_price, "size": depth}],
    )


def test_alpha_row_bounds() -> None:
    assert alpha_row(depth=20.0, size=10, alpha_base=0.6) == 0.6
    assert alpha_row(depth=2.0, size=10, alpha_base=0.6) == pytest.approx(0.12)


def test_fill_ratio_estimator_depth_awareness() -> None:
    estimator = FillRatioEstimator(alpha=0.6)

    rich_orderbook = _orderbook(0.4, depth=50.0)
    expected, ratio = estimator.estimate(
        side="YES",
        price=0.4,
        contracts=10,
        orderbook=rich_orderbook,
    )
    assert expected == 6
    assert ratio == pytest.approx(0.6, rel=1e-6)

    thin_orderbook = _orderbook(0.4, depth=2.0)
    expected_thin, ratio_thin = estimator.estimate(
        side="YES",
        price=0.4,
        contracts=10,
        orderbook=thin_orderbook,
    )
    assert expected_thin == 1
    assert ratio_thin == pytest.approx(0.1, rel=1e-6)
