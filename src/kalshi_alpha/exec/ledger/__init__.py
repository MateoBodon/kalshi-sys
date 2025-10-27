"""Paper trading ledger utilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
import csv
import json

from kalshi_alpha.core.fees import DEFAULT_FEE_SCHEDULE, FeeSchedule, maker_fee
from kalshi_alpha.core.kalshi_api import Orderbook
from kalshi_alpha.core.pricing import Liquidity
from kalshi_alpha.core.execution.slippage import SlippageModel, price_with_slippage

if TYPE_CHECKING:  # pragma: no cover
    from kalshi_alpha.exec.runners.scan_ladders import Proposal


@dataclass
class FillRecord:
    proposal: Proposal
    fill_price: float
    expected_value: float
    liquidity: Liquidity
    slippage: float


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

    def write_artifacts(self, output_dir: Path) -> tuple[Path | None, Path | None]:
        if not self.records:
            return None, None
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S")
        json_path = output_dir / f"{timestamp}_ledger.json"
        csv_path = output_dir / f"{timestamp}_ledger.csv"

        rows = [
            {
                "market_id": record.proposal.market_id,
                "strike": record.proposal.strike,
                "side": record.proposal.side,
                "contracts": record.proposal.contracts,
                "fill_price": record.fill_price,
                "expected_value": record.expected_value,
                "slippage": record.slippage,
            }
            for record in self.records
        ]

        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "market_id",
                    "strike",
                    "side",
                    "contracts",
                    "fill_price",
                    "expected_value",
                    "slippage",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        payload = {
            "summary": self.to_dict(),
            "fills": rows,
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return json_path, csv_path


def simulate_fills(
    proposals: Sequence[Proposal],
    orderbooks: dict[str, Orderbook],
    *,
    mode: str = "top",
    slippage_model: SlippageModel | None = None,
    schedule: FeeSchedule = DEFAULT_FEE_SCHEDULE,
    artifacts_dir: Path | None = None,
) -> PaperLedger:
    ledger = PaperLedger()
    model = None
    if slippage_model is not None:
        model = slippage_model
    elif mode in {"top", "depth"}:
        model = SlippageModel(mode=mode)
    for proposal in proposals:
        orderbook = orderbooks.get(proposal.market_id)
        if orderbook is None:
            continue
        fill_price, slippage = _derive_fill_price(
            proposal, orderbook, mode=mode, slippage_model=model
        )
        expected = _expected_value_with_fill(proposal, fill_price, schedule=schedule)
        ledger.record(
            FillRecord(
                proposal=proposal,
                fill_price=fill_price,
                expected_value=expected,
                liquidity=Liquidity.MAKER,
                slippage=slippage,
            )
        )
    if artifacts_dir is not None:
        ledger.write_artifacts(artifacts_dir)
    return ledger


def _derive_fill_price(
    proposal: Proposal,
    orderbook: Orderbook,
    *,
    mode: str,
    slippage_model: SlippageModel | None,
) -> tuple[float, float]:
    if slippage_model is None and mode == "mid":
        try:
            price = orderbook.depth_weighted_mid()
        except Exception:
            price = proposal.market_yes_price
        return price, price - proposal.market_yes_price

    model = slippage_model or SlippageModel(mode="top")
    price, impact = price_with_slippage(
        side=proposal.side,
        contracts=proposal.contracts,
        proposal_price=proposal.market_yes_price,
        orderbook=orderbook,
        model=model,
    )
    return price, impact


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
