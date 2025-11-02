"""Typed ledger schema definitions."""

from __future__ import annotations

import math
from datetime import datetime
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LedgerRowV1(BaseModel):
    """Canonical ledger row schema."""

    model_config = ConfigDict(extra="forbid", frozen=False, str_strip_whitespace=True)

    field_order: ClassVar[tuple[str, ...]] = (
        "series",
        "event",
        "market",
        "bin",
        "side",
        "price",
        "model_p",
        "market_p",
        "delta_p",
        "size",
        "expected_contracts",
        "expected_fills",
        "fill_ratio",
        "t_fill_ms",
        "size_partial",
        "slippage_ticks",
        "ev_expected_bps",
        "ev_realized_bps",
        "fees_bps",
        "slippage_mode",
        "impact_cap",
        "fees_maker",
        "ev_after_fees",
        "pnl_simulated",
        "timestamp_et",
        "manifest_path",
        "ledger_schema_version",
    )

    series: str
    event: str
    market: str
    bin: float
    side: str
    price: float
    model_p: float
    market_p: float
    delta_p: float
    size: int
    expected_contracts: int
    expected_fills: int
    fill_ratio: float
    t_fill_ms: float = 0.0
    size_partial: int = 0
    slippage_ticks: float = 0.0
    ev_expected_bps: float = 0.0
    ev_realized_bps: float = 0.0
    fees_bps: float = 0.0
    slippage_mode: str
    impact_cap: float
    fees_maker: float
    ev_after_fees: float
    pnl_simulated: float
    timestamp_et: datetime
    manifest_path: str = ""
    ledger_schema_version: int = Field(default=2)

    @classmethod
    def canonical_fields(cls) -> tuple[str, ...]:
        return cls.field_order

    def ordered_dict(self) -> dict[str, object]:
        data = self.model_dump(mode="json")
        return {key: data[key] for key in self.field_order}

    @field_validator("side", mode="before")
    @classmethod
    def _normalize_side(cls, value: str) -> str:
        return (value or "").upper()

    @field_validator(
        "price",
        "model_p",
        "market_p",
        "delta_p",
        "fill_ratio",
        "t_fill_ms",
        "slippage_ticks",
        "ev_expected_bps",
        "ev_realized_bps",
        "fees_bps",
        "impact_cap",
        "fees_maker",
        "ev_after_fees",
        "pnl_simulated",
        mode="before",
    )
    @classmethod
    def _ensure_finite(cls, value: object) -> object:
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                raise ValueError("ledger fields must be finite")
        return value
