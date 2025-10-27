"""Paper trading ledger utilities."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import TYPE_CHECKING

from zoneinfo import ZoneInfo

from kalshi_alpha.core.execution.fillratio import FillRatioEstimator
from kalshi_alpha.core.execution.slippage import SlippageModel, price_with_slippage
from kalshi_alpha.core.fees import DEFAULT_FEE_SCHEDULE, FeeSchedule, maker_fee
from kalshi_alpha.core.kalshi_api import Orderbook
from kalshi_alpha.core.pricing import Liquidity

from .schema import LedgerRowV1

if TYPE_CHECKING:  # pragma: no cover
    from kalshi_alpha.exec.runners.scan_ladders import Proposal

ET = ZoneInfo("America/New_York")
CENT = Decimal("0.01")


@dataclass
class FillRecord:
    proposal: Proposal
    fill_price: float
    expected_value: float
    liquidity: Liquidity
    slippage: float
    expected_contracts: int
    expected_fills: int
    fill_ratio: float
    slippage_mode: str
    impact_cap: float
    fees_maker: float
    pnl_simulated: float


@dataclass
class PaperLedger:
    records: list[FillRecord] = field(default_factory=list)
    series: str = ""
    event_lookup: dict[str, str] = field(default_factory=dict)
    manifest_path: Path | None = None

    def record(self, record: FillRecord) -> None:
        self.records.append(record)

    def set_series(self, series: str | None) -> None:
        self.series = (series or "").upper()

    def set_event_lookup(self, mapping: Mapping[str, str] | None) -> None:
        self.event_lookup = {key: value for key, value in (mapping or {}).items()}

    def set_manifest_path(self, manifest: Path | None) -> None:
        self.manifest_path = manifest

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

    def iter_rows(
        self,
        *,
        timestamp_et: datetime,
        manifest_path: Path | None = None,
    ) -> list[LedgerRowV1]:
        manifest = manifest_path or self.manifest_path
        manifest_str = manifest.as_posix() if isinstance(manifest, Path) else (str(manifest) if manifest else "")
        rows: list[LedgerRowV1] = []
        for record in self.records:
            series_label_raw = self.series or record.proposal.strategy or ""
            series_label = str(series_label_raw).upper()
            event_raw = self.event_lookup.get(record.proposal.market_id) or ""
            event_label = str(event_raw)
            row = LedgerRowV1(
                series=series_label,
                event=event_label,
                market=record.proposal.market_ticker,
                bin=float(record.proposal.strike),
                side=record.proposal.side,
                price=_round_cents(record.fill_price),
                model_p=float(record.proposal.strategy_probability),
                market_p=float(record.proposal.survival_market),
                delta_p=float(
                    record.proposal.strategy_probability - record.proposal.survival_market
                ),
                size=int(record.proposal.contracts),
                expected_contracts=int(record.expected_contracts),
                expected_fills=int(record.expected_fills),
                fill_ratio=float(record.fill_ratio),
                slippage_mode=record.slippage_mode,
                impact_cap=float(record.impact_cap),
                fees_maker=record.fees_maker,
                ev_after_fees=_round_cents(record.expected_value),
                pnl_simulated=_round_cents(record.pnl_simulated),
                timestamp_et=timestamp_et,
                manifest_path=manifest_str,
            )
            rows.append(row)
        return rows

    def write_artifacts(
        self,
        output_dir: Path,
        *,
        manifest_path: Path | None = None,
    ) -> tuple[Path | None, Path | None]:
        if not self.records:
            return None, None
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp_utc = datetime.now(tz=UTC)
        timestamp_et = timestamp_utc.astimezone(ET)
        timestamp_str = timestamp_utc.strftime("%Y%m%dT%H%M%S")
        json_path = output_dir / f"{timestamp_str}_ledger.json"
        csv_path = output_dir / f"{timestamp_str}_ledger.csv"
        if manifest_path is not None:
            self.manifest_path = manifest_path

        rows = self.iter_rows(timestamp_et=timestamp_et, manifest_path=manifest_path)
        fieldnames = list(LedgerRowV1.canonical_fields())

        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row.ordered_dict())

        payload = {
            "summary": self.to_dict(),
            "rows": [row.ordered_dict() for row in rows],
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return json_path, csv_path


def simulate_fills(
    proposals: Sequence[Proposal],
    orderbooks: Mapping[str, Orderbook],
    *,
    mode: str = "top",
    slippage_model: SlippageModel | None = None,
    schedule: FeeSchedule = DEFAULT_FEE_SCHEDULE,
    artifacts_dir: Path | None = None,
    fill_estimator: FillRatioEstimator | None = None,
    ledger_series: str | None = None,
    market_event_lookup: Mapping[str, str] | None = None,
    manifest_path: Path | None = None,
) -> PaperLedger:
    ledger = PaperLedger()
    ledger.set_series(ledger_series)
    ledger.set_event_lookup(market_event_lookup)
    if manifest_path is not None:
        ledger.set_manifest_path(manifest_path)

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
            proposal,
            orderbook,
            mode=mode,
            slippage_model=model,
        )
        size = max(0, proposal.contracts)
        expected_contracts = size
        expected_fills = size
        fill_ratio = 1.0 if size > 0 else 0.0
        if fill_estimator is not None and size > 0:
            expected_fills, fill_ratio = fill_estimator.estimate(
                side=proposal.side,
                price=fill_price,
                contracts=proposal.contracts,
                orderbook=orderbook,
            )
            expected_contracts = expected_fills
        expected_contracts = max(0, expected_contracts)
        expected_fills = max(0, expected_fills)
        if size <= 0:
            fill_ratio = 0.0
        else:
            fill_ratio = max(0.0, min(fill_ratio, 1.0))

        fees = 0.0
        if expected_contracts > 0:
            try:
                price_for_fee = fill_price if proposal.side == "YES" else 1 - fill_price
                fees_dec = maker_fee(
                    expected_contracts,
                    price_for_fee,
                    schedule=schedule,
                )
            except ValueError:
                fees_dec = Decimal("0.00")
            fees = float(fees_dec)

        expected = _expected_value_with_fill(
            proposal,
            fill_price,
            schedule=schedule,
            contracts_override=expected_contracts,
        )

        slippage_mode = model.mode if model is not None else mode
        impact_cap = float(getattr(model, "impact_cap", 0.0)) if slippage_mode != "mid" else 0.0

        ledger.record(
            FillRecord(
                proposal=proposal,
                fill_price=fill_price,
                expected_value=expected,
                liquidity=Liquidity.MAKER,
                slippage=slippage,
                expected_contracts=expected_contracts,
                expected_fills=expected_fills,
                fill_ratio=fill_ratio,
                slippage_mode=slippage_mode,
                impact_cap=impact_cap,
                fees_maker=fees,
                pnl_simulated=expected,
            )
        )

    if artifacts_dir is not None:
        ledger.write_artifacts(artifacts_dir, manifest_path=manifest_path)
    return ledger


def _round_cents(value: float | Decimal) -> float:
    if isinstance(value, Decimal):
        dec_value = value
    else:
        dec_value = Decimal(str(value))
    return float(dec_value.quantize(CENT, rounding=ROUND_HALF_UP))


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
    contracts_override: int | None = None,
) -> float:
    probability = proposal.strategy_probability
    price = fill_price
    contracts = contracts_override if contracts_override is not None else proposal.contracts
    if contracts <= 0:
        return 0.0
    if proposal.side == "YES":
        fee = float(maker_fee(contracts, price, schedule=schedule))
        payoff_win = (1 - price) * contracts
        payoff_loss = -price * contracts
        return probability * payoff_win + (1 - probability) * payoff_loss - fee
    no_price = 1 - price
    fee = float(maker_fee(contracts, no_price, schedule=schedule))
    payoff_win = (1 - no_price) * contracts
    payoff_loss = -no_price * contracts
    return (1 - probability) * payoff_win + probability * payoff_loss - fee
