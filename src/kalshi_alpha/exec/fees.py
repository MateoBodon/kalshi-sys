"""Execution-time fee helpers backed by configs/fees.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import ROUND_UP, Decimal, getcontext
from functools import lru_cache
from pathlib import Path

getcontext().prec = 18

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FEES_PATH = ROOT / "configs" / "fees.json"


@dataclass(frozen=True)
class FeeSeriesRule:
    """Per-series fee coefficients used for order-level rounding."""

    series: str
    coefficient: Decimal
    maker_fee: Decimal
    taker_fee: Decimal

    def rate_for(self, liquidity: str) -> Decimal:
        normalized = liquidity.strip().lower()
        if normalized == "maker":
            return self.maker_fee
        if normalized == "taker":
            return self.taker_fee
        raise ValueError(f"Unsupported liquidity '{liquidity}'. Expected 'maker' or 'taker'.")


@dataclass(frozen=True)
class FeeConfig:
    path: Path
    rounding_quantum: Decimal
    rounding_mode: str
    series: dict[str, FeeSeriesRule]


def _round_up(value: Decimal, quantum: Decimal, *, mode: str) -> Decimal:
    if value <= 0:
        return Decimal("0.00")
    if mode != "ROUND_UP":
        raise ValueError(f"Unsupported rounding mode '{mode}' in fee configuration")
    exponent = quantum.normalize()
    return value.quantize(exponent, rounding=ROUND_UP)


def _to_decimal(value: float | int | Decimal, *, field: str) -> Decimal:
    try:
        return Decimal(str(value))
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid numeric value for {field}: {value}") from exc


@lru_cache(maxsize=4)
def load_fee_config(path: Path | None = None) -> FeeConfig:
    """Load maker/taker fee coefficients for indices."""

    resolved = (path or DEFAULT_FEES_PATH).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Fee configuration not found at {resolved}")

    data = json.loads(resolved.read_text(encoding="utf-8"))
    rounding_section = data.get("rounding", {}) if isinstance(data, dict) else {}
    quantum = _to_decimal(rounding_section.get("quantum", "0.01"), field="rounding.quantum")
    if quantum <= 0:
        raise ValueError("rounding.quantum must be positive")
    mode = str(rounding_section.get("mode", "ROUND_UP")).strip().upper()
    series_section = data.get("series") if isinstance(data, dict) else None
    if not isinstance(series_section, dict) or not series_section:
        raise ValueError("fees configuration must contain a 'series' mapping")

    rules: dict[str, FeeSeriesRule] = {}
    for key, raw in series_section.items():
        if not isinstance(raw, dict):
            continue
        series_key = str(key).upper()
        coefficient = _to_decimal(raw.get("coefficient", "0"), field=f"series[{series_key}].coefficient")
        maker_fee = _to_decimal(raw.get("maker_fee", "0"), field=f"series[{series_key}].maker_fee")
        taker_fee = _to_decimal(raw.get("taker_fee", coefficient), field=f"series[{series_key}].taker_fee")
        rules[series_key] = FeeSeriesRule(
            series=series_key,
            coefficient=coefficient,
            maker_fee=maker_fee,
            taker_fee=taker_fee,
        )

    if not rules:
        raise ValueError("fees configuration contained no usable series entries")

    return FeeConfig(
        path=resolved,
        rounding_quantum=quantum,
        rounding_mode=mode,
        series=rules,
    )


def order_fee(
    *,
    series: str,
    price: float | Decimal,
    contracts: int | float | Decimal,
    liquidity: str,
    config: FeeConfig | None = None,
) -> Decimal:
    """Compute the rounded fee charged for a single order submission."""

    cfg = config or load_fee_config()
    series_key = series.upper()
    rule = cfg.series.get(series_key)
    if rule is None:
        raise KeyError(f"Series '{series}' missing from fee configuration {cfg.path}")
    price_dec = _to_decimal(price, field="price")
    if price_dec < 0 or price_dec > 1:
        raise ValueError("price must lie within [0, 1]")
    contracts_dec = _to_decimal(contracts, field="contracts")
    if contracts_dec <= 0:
        raise ValueError("contracts must be positive")

    rate = rule.rate_for(liquidity)
    if rate <= 0:
        return Decimal("0.00")
    raw = rate * contracts_dec * price_dec * (Decimal("1") - price_dec)
    return _round_up(raw, cfg.rounding_quantum, mode=cfg.rounding_mode)


def fee_breakdown(
    *,
    series: str,
    price: float | Decimal,
    contracts: int,
    liquidity: str = "maker",
    config: FeeConfig | None = None,
) -> dict[str, Decimal]:
    """Return per-order and per-contract fee metrics for metadata logging."""

    per_order = order_fee(
        series=series,
        price=price,
        contracts=contracts,
        liquidity=liquidity,
        config=config,
    )
    effective_per_contract = (
        per_order / Decimal(max(contracts, 1))
        if per_order > 0
        else Decimal("0.00")
    )
    return {
        "per_order": per_order,
        "per_contract_effective": effective_per_contract,
    }


__all__ = ["FeeConfig", "FeeSeriesRule", "fee_breakdown", "load_fee_config", "order_fee"]
