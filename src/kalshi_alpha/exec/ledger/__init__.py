"""Paper trading ledger utilities."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from kalshi_alpha.core.execution.fillratio import FillRatioEstimator, _visible_depth, alpha_row
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
    alpha_row: float = 0.0
    size_throttled: bool = False
    t_fill_ms: float = 0.0
    size_partial: int = 0
    slippage_ticks: float = 0.0
    ev_expected_bps: float = 0.0
    ev_realized_bps: float = 0.0
    fees_bps: float = 0.0


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
        if manifest is None:
            manifest_str = ""
        elif isinstance(manifest, Path):
            manifest_str = manifest.as_posix()
        else:
            manifest_str = str(manifest)
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
                t_fill_ms=float(record.t_fill_ms),
                size_partial=int(record.size_partial),
                slippage_ticks=float(record.slippage_ticks),
                ev_expected_bps=float(record.ev_expected_bps),
                ev_realized_bps=float(record.ev_realized_bps),
                fees_bps=float(record.fees_bps),
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


def simulate_fills(  # noqa: PLR0913, PLR0912, PLR0915
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
        alpha_row_value = 1.0 if size > 0 else 0.0
        size_throttled = False

        alpha_row_base: float | None = None
        visible_depth: float | None = None
        if fill_estimator is not None and size > 0:
            visible_depth = _visible_depth(proposal.side, fill_price, orderbook)
            alpha_row_base = alpha_row(visible_depth, size, fill_estimator.alpha)
            alpha_row_base = max(0.0, min(alpha_row_base, 1.0))
            expected_contracts = int(math.floor(size * alpha_row_base))
            expected_fills = max(0, expected_contracts)
        if size > 0:
            depth_limited = visible_depth is not None and visible_depth < size
            throttle_reference = 0.7
            if (
                alpha_row_base is not None
                and alpha_row_base < throttle_reference
                and expected_contracts > 0
                and depth_limited
            ):
                scale_factor = (
                    alpha_row_base / throttle_reference if throttle_reference > 0 else 0.0
                )
                throttled_contracts = int(math.floor(expected_contracts * scale_factor))
                if throttled_contracts < expected_contracts:
                    expected_contracts = max(0, throttled_contracts)
                    expected_fills = expected_contracts
                    size_throttled = True
            fill_ratio = expected_fills / size if size > 0 else 0.0
            if alpha_row_base is None:
                alpha_row_value = 1.0
            else:
                alpha_row_value = fill_ratio
        else:
            fill_ratio = 0.0
            alpha_row_value = 0.0

        expected_contracts = max(0, expected_contracts)
        expected_fills = max(0, expected_fills)
        fill_ratio = max(0.0, min(fill_ratio, 1.0))

        fees = 0.0
        if expected_contracts > 0:
            try:
                price_for_fee = fill_price if proposal.side == "YES" else 1 - fill_price
                fees_dec = maker_fee(
                    expected_contracts,
                    price_for_fee,
                    series=proposal.series,
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

        notional = max(float(size) * float(fill_price), 1e-9)
        partial_contracts = max(0, size - expected_fills)
        slippage_ticks = float(slippage) / float(CENT) if CENT != 0 else 0.0
        expected_bps = float(proposal.maker_ev_per_contract) * 10000.0 if size > 0 else 0.0
        realized_bps = (expected / notional) * 10000.0 if notional > 0 else 0.0
        fees_bps = (fees / notional) * 10000.0 if notional > 0 else 0.0

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
                alpha_row=alpha_row_value,
                size_throttled=size_throttled,
                t_fill_ms=0.0,
                size_partial=partial_contracts,
                slippage_ticks=slippage_ticks,
                ev_expected_bps=expected_bps,
                ev_realized_bps=realized_bps,
                fees_bps=fees_bps,
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
        fee = float(
            maker_fee(
                contracts,
                price,
                series=proposal.series if hasattr(proposal, "series") else None,
                schedule=schedule,
            )
        )
        payoff_win = (1 - price) * contracts
        payoff_loss = -price * contracts
        return probability * payoff_win + (1 - probability) * payoff_loss - fee
    no_price = 1 - price
    fee = float(
        maker_fee(
            contracts,
            no_price,
            series=proposal.series if hasattr(proposal, "series") else None,
            schedule=schedule,
        )
    )
    payoff_win = (1 - no_price) * contracts
    payoff_loss = -no_price * contracts
    return (1 - probability) * payoff_win + probability * payoff_loss - fee
