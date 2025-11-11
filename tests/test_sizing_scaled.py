from __future__ import annotations

from kalshi_alpha.core.sizing.kelly import scale_kelly
from kalshi_alpha.exec.limits import LossBudget


def test_scale_kelly_applies_penalties() -> None:
    base = 0.2
    scaled = scale_kelly(base, uncertainty=0.5, ob_imbalance=0.25, cap=0.25)
    expected = 0.2 * (1 - 0.5) * (1 - 0.25)
    assert scaled == expected


def test_scale_kelly_caps_value() -> None:
    scaled = scale_kelly(0.5, uncertainty=0.0, ob_imbalance=0.0, cap=0.25)
    assert scaled == 0.25
    negative = scale_kelly(-0.5, uncertainty=0.0, ob_imbalance=0.0, cap=0.25)
    assert negative == -0.25


def test_loss_budget_allocation_and_exhaustion() -> None:
    budget = LossBudget(100.0)
    assert budget.max_contracts(25.0, 10) == 4
    budget.consume(60.0)
    assert budget.remaining == 40.0
    assert budget.max_contracts(10.0, 10) == 4
    budget.consume(35.0)
    assert budget.remaining == 5.0
    assert budget.max_contracts(10.0, 10) == 0


def test_loss_budget_unbounded() -> None:
    budget = LossBudget(None)
    assert budget.max_contracts(100.0, 5) == 5
    budget.consume(500.0)
    assert budget.remaining is None
