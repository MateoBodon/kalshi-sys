import pytest

from kalshi_alpha.core.pricing import Liquidity, OrderSide
from kalshi_alpha.core.risk import OrderProposal, PALGuard, PALPolicy, max_loss_for_order


def test_max_loss_includes_taker_fee() -> None:
    order = OrderProposal(
        strike_id="CPI-2025-10:0.35",
        yes_price=0.35,
        contracts=1,
        side=OrderSide.YES,
        liquidity=Liquidity.TAKER,
    )
    loss = max_loss_for_order(order)
    assert pytest.approx(loss, abs=1e-9) == 0.37  # 0.35 price + 0.02 fee


def test_pal_guard_limit_enforcement() -> None:
    policy = PALPolicy(series="CPI", default_max_loss=100.0)
    guard = PALGuard(policy)

    small_order = OrderProposal(
        strike_id="CPI-2025-10:0.35",
        yes_price=0.5,
        contracts=100,
        side=OrderSide.YES,
        liquidity=Liquidity.MAKER,
    )
    assert guard.can_accept(small_order)
    guard.register(small_order)

    large_order = OrderProposal(
        strike_id="CPI-2025-10:0.35",
        yes_price=0.5,
        contracts=300,
        side=OrderSide.YES,
        liquidity=Liquidity.MAKER,
    )
    assert not guard.can_accept(large_order)
    with pytest.raises(ValueError):
        guard.register(large_order)
