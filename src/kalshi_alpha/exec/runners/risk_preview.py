"""CLI to preview risk posture before running a ladder scan."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Iterable

import polars as pl

from kalshi_alpha.core.gates import QualityGateResult, load_quality_gate_config, run_quality_gates
from kalshi_alpha.core.risk import PALPolicy, PortfolioConfig, PortfolioRiskManager
from kalshi_alpha.exec.pipelines.daily import resolve_series

DEFAULT_PORTFOLIO_CONFIG = Path("configs/portfolio.yaml")
DEFAULT_QUALITY_GATES = Path("configs/quality_gates.yaml")
DEFAULT_PAL_POLICY = Path("configs/pal_policy.yaml")
LEDGER_PATH = Path("data/proc/ledger_all.parquet")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview risk and gate status before executing a scan.")
    parser.add_argument("--mode", required=True, help="Pipeline mode (e.g. pre_cpi, pre_claims).")
    parser.add_argument(
        "--when",
        type=_parse_date,
        help="Target date in YYYY-MM-DD (defaults to today ET).",
    )
    parser.add_argument("--offline", action="store_true", help="Reserved for parity with other CLIs.")
    parser.add_argument("--portfolio-config", type=Path, default=DEFAULT_PORTFOLIO_CONFIG)
    parser.add_argument("--quality-gates-config", type=Path, default=DEFAULT_QUALITY_GATES)
    parser.add_argument("--pal-policy", type=Path, default=DEFAULT_PAL_POLICY)
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    target_date = args.when or datetime.now(tz=UTC).date()
    series = resolve_series(args.mode)
    if series is None:
        print(f"[risk_preview] Unable to resolve series for mode={args.mode}")
        raise SystemExit(1)

    print(f"[risk_preview] mode={args.mode} date={target_date.isoformat()} series={series}")

    portfolio_config = _load_portfolio_config(args.portfolio_config)
    pal_policy = _load_pal_policy(args.pal_policy)
    ledger = _load_ledger(args.ledger, series)

    factor_exposures, per_strike_loss, net_brackets = _compute_exposures(ledger, portfolio_config, series)
    var_estimate = _estimate_var(portfolio_config, factor_exposures)
    print(f"[risk_preview] Approx VaR={var_estimate:.2f}")

    if per_strike_loss:
        print("[risk_preview] Per-strike max loss projections:")
        for strike, value in sorted(per_strike_loss.items()):
            print(f"  - {strike}: {value:.2f} USD")

    pal_limits = _pal_limits_summary(pal_policy)
    if pal_limits:
        print("[risk_preview] PAL limits:")
        for strike, cap in pal_limits.items():
            print(f"  - {strike}: {cap:.2f} USD")

    if net_brackets:
        print("[risk_preview] Net ladder exposure (contracts):")
        for rng, net in sorted(net_brackets.items()):
            print(f"  - {rng}: {net:+.0f}")

    if factor_exposures:
        print("[risk_preview] Factor loads (USD):")
        for factor, exposure in sorted(factor_exposures.items()):
            print(f"  - {factor}: {exposure:.2f}")

    gate_result = _run_quality_gates(args.quality_gates_config)
    _print_gate_result(gate_result)
    if not gate_result.go:
        raise SystemExit(1)


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _load_portfolio_config(path: Path) -> PortfolioConfig | None:
    if path.exists():
        return PortfolioConfig.from_yaml(path)
    example = path.with_suffix(".example.yaml")
    if example.exists():
        return PortfolioConfig.from_yaml(example)
    return None


def _load_pal_policy(path: Path) -> PALPolicy | None:
    if path.exists():
        return PALPolicy.from_yaml(path)
    example = path.with_suffix(".example.yaml")
    if example.exists():
        return PALPolicy.from_yaml(example)
    return None


def _load_ledger(path: Path, series: str) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame()
    frame = pl.read_parquet(path)
    if "timestamp_et" in frame.columns and frame["timestamp_et"].dtype == pl.Utf8:
        frame = frame.with_columns(pl.col("timestamp_et").str.strptime(pl.Datetime, strict=False))
    return frame.filter(pl.col("series") == series.upper())


def _compute_exposures(
    ledger: pl.DataFrame,
    portfolio_config: PortfolioConfig | None,
    series: str,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    if ledger.is_empty():
        return {}, {}, {}
    factor_exposures: dict[str, float] = defaultdict(float)
    per_strike_loss: dict[str, float] = defaultdict(float)
    ladder_net: dict[str, float] = defaultdict(float)
    betas = {}
    if portfolio_config is not None:
        betas = portfolio_config.strategy_betas.get(series.upper(), {})
    if not betas:
        betas = {"TOTAL": 1.0}
    for row in ledger.to_dicts():
        contracts = float(row.get("size") or 0.0)
        strike = f"{row.get('market')}@{row.get('bin')}"
        side = str(row.get("side", "YES")).upper()
        price = float(row.get("price") or 0.0)
        fees = float(row.get("fees_maker") or 0.0)
        if contracts <= 0:
            continue
        if side == "YES":
            max_loss = contracts * price + fees
            ladder_net[strike] += contracts
        else:
            max_loss = contracts * (1.0 - price) + fees
            ladder_net[strike] -= contracts
        per_strike_loss[strike] += max_loss
        for factor, beta in betas.items():
            factor_exposures[factor] += beta * max_loss
    return factor_exposures, per_strike_loss, ladder_net


def _estimate_var(config: PortfolioConfig | None, exposures: dict[str, float]) -> float:
    if not exposures:
        return 0.0
    if config is None:
        return sum(abs(val) for val in exposures.values())
    manager = PortfolioRiskManager(config)
    return manager._compute_var(exposures)  # type: ignore[attr-defined]


def _pal_limits_summary(policy: PALPolicy | None) -> dict[str, float]:
    if policy is None:
        return {}
    summary = dict(policy.per_strike)
    if policy.default_max_loss:
        summary.setdefault("DEFAULT", float(policy.default_max_loss))
    return summary


def _run_quality_gates(config_path: Path) -> QualityGateResult:
    if not config_path.exists():
        fallback = config_path.with_suffix(".example.yaml") if not str(config_path).endswith(".example.yaml") else None
        cfg = load_quality_gate_config(fallback)
    else:
        cfg = load_quality_gate_config(config_path)
    result = run_quality_gates(config=cfg, now=datetime.now(tz=UTC))
    return result


def _print_gate_result(result: QualityGateResult) -> None:
    status = "GO" if result.go else "NO-GO"
    print(f"[risk_preview] Quality gates verdict: {status}")
    if result.reasons:
        print("[risk_preview] Reasons:")
        for reason in result.reasons:
            print(f"  - {reason}")
    if result.details:
        print("[risk_preview] Details:")
        for key, value in result.details.items():
            print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()
