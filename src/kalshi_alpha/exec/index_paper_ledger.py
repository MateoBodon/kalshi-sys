"""Lightweight JSONL ledger for index paper (dry) trades."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping
from zoneinfo import ZoneInfo

from kalshi_alpha.datastore.paths import PROC_ROOT

ET = ZoneInfo("America/New_York")
INDEX_SERIES = frozenset({"INX", "INXU", "NASDAQ100", "NASDAQ100U"})
LEDGER_ENV_KEY = "KALSHI_INDEX_PAPER_LEDGER_PATH"
DEFAULT_LEDGER_PATH = PROC_ROOT / "ledger" / "index_paper.jsonl"
REQUIRED_FIELDS = (
    "series",
    "kalshi_market_id",
    "strike",
    "side",
    "price",
    "size",
    "ev_after_fees_cents",
)


def _resolve_path(ledger_path: Path | None = None) -> Path:
    override = os.getenv(LEDGER_ENV_KEY)
    if ledger_path is None and override:
        ledger_path = Path(override)
    target = ledger_path or DEFAULT_LEDGER_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _normalize_timestamp(value: object | None) -> str:
    if value is None:
        timestamp = datetime.now(tz=ET)
    elif isinstance(value, datetime):
        stamp = value
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=UTC)
        timestamp = stamp.astimezone(ET)
    else:
        try:
            parsed = datetime.fromisoformat(str(value))
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError("timestamp_et must be a datetime or ISO-8601 string") from exc
        if parsed.tzinfo is None:
            timestamp = parsed.replace(tzinfo=ET)
        else:
            timestamp = parsed.astimezone(ET)
    return timestamp.isoformat()


def _normalize_side(value: object | None) -> str:
    raw = (str(value or "")).strip().lower()
    if raw in {"yes", "buy", "b"}:
        return "buy"
    if raw in {"no", "sell", "s"}:
        return "sell"
    raise ValueError(f"Unsupported side '{value}' for index paper ledger")


def _coerce_float(field: str, value: object | None) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"{field} must be numeric for index paper ledger") from exc


def log_index_paper_trade(
    record: Mapping[str, object],
    *,
    ledger_path: Path | None = None,
) -> Path:
    """Append a single index paper trade to the JSONL ledger."""

    payload = dict(record)
    series = str(payload.get("series") or "").strip().upper()
    if series not in INDEX_SERIES:
        raise ValueError(f"Series '{series}' is not an index ladder series")
    payload["series"] = series

    market_id = (
        payload.get("kalshi_market_id")
        or payload.get("market_id")
        or payload.get("market")
        or payload.get("market_ticker")
    )
    market_id_str = str(market_id or "").strip()
    if not market_id_str:
        raise ValueError("kalshi_market_id is required for index paper ledger")
    payload["kalshi_market_id"] = market_id_str

    for required in REQUIRED_FIELDS:
        if required not in payload:
            raise ValueError(f"Missing required field '{required}' for index paper ledger")

    payload["timestamp_et"] = _normalize_timestamp(payload.get("timestamp_et"))
    payload["strike"] = _coerce_float("strike", payload.get("strike"))
    payload["price"] = _coerce_float("price", payload.get("price"))
    payload["size"] = int(payload.get("size") or 0)
    payload["ev_after_fees_cents"] = _coerce_float(
        "ev_after_fees_cents",
        payload.get("ev_after_fees_cents"),
    )
    fill_prob = payload.get("fill_prob")
    payload["fill_prob"] = None if fill_prob is None else _coerce_float("fill_prob", fill_prob)
    payload["side"] = _normalize_side(payload.get("side"))
    payload["mode"] = "dry"
    window = payload.get("window")
    payload["window"] = str(window) if window else None

    target = _resolve_path(ledger_path)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")
    return target


__all__ = ["DEFAULT_LEDGER_PATH", "INDEX_SERIES", "LEDGER_ENV_KEY", "log_index_paper_trade"]
