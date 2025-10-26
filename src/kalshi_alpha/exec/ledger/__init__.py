"""Paper trading ledger utilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from kalshi_alpha.core.fees import DEFAULT_FEE_SCHEDULE, FeeSchedule, maker_fee
from kalshi_alpha.core.kalshi_api import Orderbook
from kalshi_alpha.core.pricing import Liquidity

if TYPE_CHECKING:  # pragma: no cover
    from kalshi_alpha.exec.runners.scan_ladders import Proposal


@dataclass
class FillRecord:
    proposal: Proposal
    fill_price: float
    expected_value: float
    liquidity: Liquidity


@dataclass
class PaperLedger:
    records: list[FillRecord] = field(default_factory=list)

    def record(self, record: FillRecord) -> None:
        self.records.append(record)

    def total_expected_pnl(self) -> float:
        return sum(record.expected_value for record in self.records)

    def total_max_loss(self) -> float:
        return sum(record.proposal.max_loss for record in self.records)

    def to_dict(self) -> dict[str, float]:
        return {
            "expected_pnl": self.total_expected_pnl(),
            "max_loss": self.total_max_loss(),
            "trades": len(self.records),
        }


def simulate_fills(
    proposals: Sequence[Proposal],
    orderbooks: dict[str, Orderbook],
    *,
    mode: str = "top",
    schedule: FeeSchedule = DEFAULT_FEE_SCHEDULE,
) -> PaperLedger:
    ledger = PaperLedger()
    for proposal in proposals:
        orderbook = orderbooks.get(proposal.market_id)
        if orderbook is None:
            continue
        fill_price = _derive_fill_price(proposal, orderbook, mode=mode)
        expected = _expected_value_with_fill(proposal, fill_price, schedule=schedule)
        ledger.record(
            FillRecord(
                proposal=proposal,
                fill_price=fill_price,
                expected_value=expected,
                liquidity=Liquidity.MAKER,
            )
        )
    return ledger


def _derive_fill_price(proposal: Proposal, orderbook: Orderbook, *, mode: str) -> float:
    if mode == "mid":
        try:
            return orderbook.depth_weighted_mid()
        except Exception:
            return proposal.market_yes_price
    if proposal.side == "YES":
        return orderbook.asks[0]["price"] if orderbook.asks else proposal.market_yes_price
    return orderbook.bids[0]["price"] if orderbook.bids else 1.0 - proposal.market_yes_price


def _expected_value_with_fill(
    proposal: Proposal,
    fill_price: float,
    *,
    schedule: FeeSchedule,
) -> float:
    probability = proposal.strategy_probability
    price = fill_price
    contracts = proposal.contracts
    if proposal.side == "YES":
        fee = float(maker_fee(contracts, price, schedule=schedule))
        payoff_win = (1 - price) * contracts
        payoff_loss = -price * contracts
        return probability * payoff_win + (1 - probability) * payoff_loss - fee
    else:
        no_price = 1 - price
        fee = float(maker_fee(contracts, no_price, schedule=schedule))
        payoff_win = (1 - no_price) * contracts
        payoff_loss = -no_price * contracts
        return (1 - probability) * payoff_win + probability * payoff_loss - fee
