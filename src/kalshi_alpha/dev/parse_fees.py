"""Parse Kalshi fee schedule PDF and emit normalized JSON configuration."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from kalshi_alpha.core import fees

PDF_DEFAULT_PATH = fees.ROOT / "docs" / "kalshi-fee-schedule.pdf"
OUTPUT_DEFAULT_PATH = fees.ROOT / "data" / "proc" / "state" / "fees.json"

MAKER_PATTERN = re.compile(r"(?i)maker[^0-9]*([0-9]+(?:\.[0-9]+)?)")
TAKER_PATTERN = re.compile(r"(?i)taker[^0-9]*([0-9]+(?:\.[0-9]+)?)")
DATE_PATTERN = re.compile(r"(?i)effective[^0-9]*([0-9]{4}-[0-9]{2}-[0-9]{2})")
SERIES_PATTERN = re.compile(r"(?i)half\s+rate\s+series[:\-]\s*(.+)")
KEYWORD_PATTERN = re.compile(r"(?i)half\s+rate\s+keywords[:\-]\s*(.+)")
BRACKET_PATTERN = re.compile(
    r"(?m)^\s*\$([0-9,]+)\s*(?:-|to|–)\s*\$?(\+|[0-9,]+)\s*[^0-9%]*([0-9]+(?:\.[0-9]+)?%?)",
    re.IGNORECASE,
)


def _normalize_text(pdf_bytes: bytes) -> str:
    """Decode PDF bytes into a best-effort string suitable for regex parsing."""

    return pdf_bytes.decode("latin-1", errors="ignore")


def _extract_rate(pattern: re.Pattern[str], text: str, fallback: float) -> float:
    match = pattern.search(text)
    if match:
        value = match.group(1).rstrip("%")
        try:
            rate = float(value)
        except ValueError:  # pragma: no cover - defensive
            return fallback
        return rate / 100.0 if match.group(1).endswith("%") else rate
    return fallback


def _extract_list(pattern: re.Pattern[str], text: str) -> list[str]:
    match = pattern.search(text)
    if not match:
        return []
    payload = match.group(1)
    tokens = re.split(r"[,\n]\s*", payload)
    return [token.strip().upper() for token in tokens if token.strip()]


def _to_amount(value: str) -> int | None:
    token = value.strip().replace(",", "")
    if token in {"+", "PLUS", "∞"}:
        return None
    try:
        return int(token)
    except ValueError:
        return None


def _parse_brackets(text: str) -> list[dict[str, Any]]:
    brackets: list[dict[str, Any]] = []
    for match in BRACKET_PATTERN.finditer(text):
        lower_raw, upper_raw, rate_raw = match.groups()
        lower = _to_amount(lower_raw)
        upper = _to_amount(upper_raw)
        rate_value = rate_raw.rstrip("%")
        try:
            rate = float(rate_value)
        except ValueError:  # pragma: no cover - defensive
            continue
        if rate_raw.endswith("%"):
            rate = rate / 100.0
        bracket = {
            "lower": lower or 0,
            "upper": upper,
            "rate": rate,
        }
        brackets.append(bracket)
    # Deduplicate and sort by lower bound
    unique: dict[tuple[int, int | None], dict[str, Any]] = {}
    for bracket in brackets:
        key = (bracket["lower"], bracket["upper"])
        unique[key] = bracket
    ordered = sorted(unique.values(), key=lambda item: item["lower"])
    return ordered


def parse_fee_schedule(pdf_path: Path, *, base_config: dict[str, Any] | None = None) -> dict[str, Any]:
    text = _normalize_text(pdf_path.read_bytes())
    config = dict(base_config or fees._load_fee_config(fees.FEE_CONFIG_PATH))

    config["maker_rate"] = _extract_rate(MAKER_PATTERN, text, float(config["maker_rate"]))
    config["taker_rate"] = _extract_rate(TAKER_PATTERN, text, float(config["taker_rate"]))

    date_match = DATE_PATTERN.search(text)
    if date_match:
        config["effective_date"] = date_match.group(1)

    series = _extract_list(SERIES_PATTERN, text)
    if series:
        config["series_half_rate"] = series

    keywords = _extract_list(KEYWORD_PATTERN, text)
    if keywords:
        config["half_rate_keywords"] = keywords

    brackets = _parse_brackets(text)
    if brackets:
        config["fee_brackets"] = brackets

    return config


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Parse Kalshi fee schedule PDF into JSON format.")
    parser.add_argument("--pdf", type=Path, default=PDF_DEFAULT_PATH, help="Input PDF path.")
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT_PATH, help="Output JSON path.")
    parser.add_argument("--quiet", action="store_true", help="Suppress status output.")
    args = parser.parse_args(argv)

    if not args.pdf.exists():
        config = fees._load_fee_config(fees.FEE_CONFIG_PATH)
    else:
        base_config = fees._load_fee_config(fees.FEE_CONFIG_PATH)
        config = parse_fee_schedule(args.pdf, base_config=base_config)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    if not args.quiet:
        print(f"Wrote Kalshi fee schedule to {output_path}")
    return output_path


if __name__ == "__main__":  # pragma: no cover
    main()
