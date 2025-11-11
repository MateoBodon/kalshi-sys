"""Limit enforcement helpers for proposal generation and broker validation."""

from __future__ import annotations

from dataclasses import dataclass

from kalshi_alpha.core.risk import OrderProposal, PALGuard, max_loss_for_order


class LimitViolation(RuntimeError):
    """Raised when a proposal exceeds PAL or loss budgets."""


@dataclass
class LossBudget:
    cap: float | None

    def __post_init__(self) -> None:
        self.remaining: float | None = float(self.cap) if self.cap and self.cap > 0 else None

    def max_contracts(self, unit_loss: float, requested: int) -> int:
        if self.cap is None or unit_loss <= 0 or requested <= 0:
            return requested
        if self.remaining is None or self.remaining <= 0:
            return 0
        allowed = int(self.remaining // unit_loss)
        return min(requested, allowed)

    def consume(self, loss: float) -> None:
        if self.cap is None or loss <= 0:
            return
        if self.remaining is None:
            self.remaining = float(self.cap)
        self.remaining = max(0.0, self.remaining - loss)


class ProposalLimitChecker:
    """Applies PAL and stop-loss budgets before proposals are admitted."""

    def __init__(
        self,
        pal_guard: PALGuard,
        *,
        daily_budget: LossBudget | None = None,
        weekly_budget: LossBudget | None = None,
    ) -> None:
        self._pal_guard = pal_guard
        self._daily = daily_budget
        self._weekly = weekly_budget

    def try_accept(self, order: OrderProposal, *, max_loss: float | None = None) -> float:
        """Validate *order* and register exposure. Returns total max loss."""

        if max_loss is None:
            max_loss = max_loss_for_order(order)
        if max_loss <= 0.0:
            raise LimitViolation("max_loss_not_positive")
        if not self._pal_guard.can_accept(order):
            raise LimitViolation("pal_exceeded")
        per_contract = max_loss / max(order.contracts, 1)
        if self._daily and self._daily.max_contracts(per_contract, order.contracts) < order.contracts:
            raise LimitViolation("daily_loss_cap_exceeded")
        if self._weekly and self._weekly.max_contracts(per_contract, order.contracts) < order.contracts:
            raise LimitViolation("weekly_loss_cap_exceeded")
        self._pal_guard.register(order)
        if self._daily:
            self._daily.consume(max_loss)
        if self._weekly:
            self._weekly.consume(max_loss)
        return max_loss


__all__ = ["LossBudget", "LimitViolation", "ProposalLimitChecker"]
