from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from kalshi_alpha.core.kalshi_api import KalshiPublicClient
from kalshi_alpha.core.risk import PALGuard, PALPolicy, PortfolioConfig, PortfolioRiskManager
from kalshi_alpha.drivers.aaa_gas import fetch as aaa_fetch
from kalshi_alpha.exec.ledger import simulate_fills
from kalshi_alpha.exec.reports import write_markdown_report
from kalshi_alpha.exec.runners.scan_ladders import (
    _attach_series_metadata,
    _strategy_pmf_for_series,
    scan_series,
    write_proposals,
)
from kalshi_alpha.exec.scanners import cpi


def test_scanner_generates_positive_maker_ev(
    fixtures_root: Path, offline_fixtures_root: Path, tmp_path: Path
) -> None:
    client = KalshiPublicClient(offline_dir=fixtures_root / "kalshi", use_offline=True)
    policy = PALPolicy(series="CPI", default_max_loss=10_000.0)
    guard = PALGuard(policy)

    outcome = scan_series(
        series="CPI",
        client=client,
        min_ev=0.01,
        contracts=5,
        pal_guard=guard,
        driver_fixtures=offline_fixtures_root,
        strategy_name="auto",
        maker_only=True,
        allow_tails=False,
        risk_manager=None,
        max_var=None,
        offline=True,
        sizing_mode="fixed",
        kelly_cap=0.25,
    )
    proposals = outcome.proposals
    assert proposals, "Expected scanner to produce proposals with positive EV"
    original_daily = aaa_fetch.DAILY_PATH
    original_monthly = aaa_fetch.MONTHLY_PATH
    try:
        aaa_fetch.DAILY_PATH = tmp_path / "aaa_daily.parquet"
        aaa_fetch.MONTHLY_PATH = tmp_path / "aaa_monthly.parquet"
        _attach_series_metadata(
            proposals=proposals,
            series="CPI",
            driver_fixtures=offline_fixtures_root,
            offline=True,
        )
    finally:
        aaa_fetch.DAILY_PATH = original_daily
        aaa_fetch.MONTHLY_PATH = original_monthly

    for proposal in proposals:
        assert proposal.taker_ev == 0.0
        assert proposal.maker_ev_per_contract >= 0.01
        assert proposal.strategy == "CPI"
        assert proposal.metadata is not None and "aaa" in proposal.metadata
        assert "aaa_mtd_average" in proposal.metadata["aaa"]

    exposures = guard.exposure_snapshot()
    assert all(value <= guard.policy.default_max_loss + 1e-6 for value in exposures.values())

    ladder = client.get_markets(client.get_events(client.get_series()[0].id)[0].id)[0]
    strikes = ladder.ladder_strikes
    strategy_pmf = cpi.strategy_pmf(strikes, fixtures_dir=offline_fixtures_root, offline=True)
    max_prob = max(bin_prob.probability for bin_prob in strategy_pmf[: len(strikes)])
    mode_indices = [
        idx
        for idx, bin_prob in enumerate(strategy_pmf[: len(strikes)])
        if bin_prob.probability >= max_prob - 1e-9
    ]
    allowed_indices: set[int] = set()
    for idx in mode_indices:
        allowed_indices.update(range(max(idx - 1, 0), min(idx + 1, len(strikes) - 1) + 1))
    strike_to_index = {float(strike): index for index, strike in enumerate(strikes)}
    for proposal in proposals:
        assert strike_to_index[proposal.strike] in allowed_indices

    output_path = write_proposals(series="CPI", proposals=proposals, output_dir=tmp_path)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["series"] == "CPI"
    assert payload["proposals"]


def test_scan_series_respects_var(
    fixtures_root: Path, offline_fixtures_root: Path, tmp_path: Path
) -> None:
    client = KalshiPublicClient(offline_dir=fixtures_root / "kalshi", use_offline=True)
    policy = PALPolicy(series="CPI", default_max_loss=1_000.0)
    guard = PALGuard(policy)
    config = PortfolioConfig(factor_vols={"TOTAL": 1.0}, strategy_betas={"CPI": {"TOTAL": 1.0}})
    manager = PortfolioRiskManager(config)

    outcome = scan_series(
        series="CPI",
        client=client,
        min_ev=0.01,
        contracts=5,
        pal_guard=guard,
        driver_fixtures=offline_fixtures_root,
        strategy_name="auto",
        maker_only=True,
        allow_tails=False,
        risk_manager=manager,
        max_var=100.0,
        offline=True,
        sizing_mode="fixed",
        kelly_cap=0.25,
    )
    proposals = outcome.proposals
    assert proposals

    manager.reset()
    guard = PALGuard(policy)
    outcome_small = scan_series(
        series="CPI",
        client=client,
        min_ev=0.01,
        contracts=5,
        pal_guard=guard,
        driver_fixtures=offline_fixtures_root,
        strategy_name="auto",
        maker_only=True,
        allow_tails=False,
        risk_manager=PortfolioRiskManager(config),
        max_var=0.1,
        offline=True,
        sizing_mode="fixed",
        kelly_cap=0.25,
    )
    assert not outcome_small.proposals


def test_paper_ledger_simulation(
    fixtures_root: Path, offline_fixtures_root: Path, tmp_path: Path
) -> None:
    client = KalshiPublicClient(offline_dir=fixtures_root / "kalshi", use_offline=True)
    policy = PALPolicy(series="CPI", default_max_loss=10_000.0)
    guard = PALGuard(policy)
    outcome = scan_series(
        series="CPI",
        client=client,
        min_ev=0.01,
        contracts=3,
        pal_guard=guard,
        driver_fixtures=offline_fixtures_root,
        strategy_name="auto",
        maker_only=True,
        allow_tails=False,
        risk_manager=None,
        max_var=None,
        offline=True,
        sizing_mode="fixed",
        kelly_cap=0.25,
    )
    proposals = outcome.proposals
    orderbooks = {
        proposal.market_id: client.get_orderbook(proposal.market_id) for proposal in proposals
    }
    ledger = simulate_fills(proposals, orderbooks)
    assert ledger.total_expected_pnl() != 0
    report_path = write_markdown_report(
        series="CPI",
        proposals=proposals,
        ledger=ledger,
        output_dir=tmp_path,
        monitors={},
    )
    assert report_path.exists()
    contents = report_path.read_text(encoding="utf-8")
    assert "| Strike | Side | Contracts |" in contents
    assert "Paper Ledger Summary" in contents


def test_kelly_sizing_respects_caps(
    fixtures_root: Path, offline_fixtures_root: Path
) -> None:
    client = KalshiPublicClient(offline_dir=fixtures_root / "kalshi", use_offline=True)
    policy = PALPolicy(series="CPI", default_max_loss=5_000.0)
    guard_fixed = PALGuard(policy)
    guard_kelly = PALGuard(policy)

    base_kwargs = dict(
        series="CPI",
        client=client,
        min_ev=0.01,
        contracts=10,
        driver_fixtures=offline_fixtures_root,
        strategy_name="auto",
        maker_only=True,
        allow_tails=False,
        risk_manager=None,
        max_var=None,
        offline=True,
    )

    fixed = scan_series(pal_guard=guard_fixed, sizing_mode="fixed", kelly_cap=0.25, **base_kwargs)
    kelly = scan_series(pal_guard=guard_kelly, sizing_mode="kelly", kelly_cap=0.1, **base_kwargs)

    assert fixed.proposals and kelly.proposals
    assert all(proposal.contracts <= base_kwargs["contracts"] for proposal in kelly.proposals)

    fixed_map = {
        (proposal.market_id, proposal.strike, proposal.side): proposal.contracts for proposal in fixed.proposals
    }
    assert all(
        proposal.contracts
        <= fixed_map.get((proposal.market_id, proposal.strike, proposal.side), base_kwargs["contracts"])
        for proposal in kelly.proposals
    )


def test_strategy_router_claims_pmf(offline_fixtures_root: Path) -> None:
    strikes = [200_000, 205_000, 210_000]
    pmf = _strategy_pmf_for_series(
        series="CLAIMS",
        strikes=strikes,
        fixtures_dir=offline_fixtures_root,
        override="auto",
        offline=True,
    )
    assert len(pmf) >= 1
    assert abs(sum(entry.probability for entry in pmf) - 1.0) < 1e-6


def test_strategy_router_teny_pmf(offline_fixtures_root: Path) -> None:
    strikes = [4.2, 4.3, 4.4]
    pmf = _strategy_pmf_for_series(
        series="TNEY",
        strikes=strikes,
        fixtures_dir=offline_fixtures_root,
        override="auto",
        offline=True,
    )
    assert len(pmf) >= len(strikes)
    assert abs(sum(entry.probability for entry in pmf) - 1.0) < 1e-6


def test_strategy_router_weather_pmf(offline_fixtures_root: Path) -> None:
    strikes = [55, 60, 65]
    pmf = _strategy_pmf_for_series(
        series="WEATHER",
        strikes=strikes,
        fixtures_dir=offline_fixtures_root,
        override="auto",
        offline=True,
    )
    assert len(pmf) >= len(strikes)
    assert abs(sum(entry.probability for entry in pmf) - 1.0) < 1e-6


def test_scan_cli_offline_smoke(tmp_path: Path, fixtures_root: Path) -> None:
    output_dir = tmp_path / "proposals"
    cmd = [
        sys.executable,
        "-m",
        "kalshi_alpha.exec.runners.scan_ladders",
        "--series",
        "CPI",
        "--offline",
        "--fixtures-root",
        str(fixtures_root),
        "--output-dir",
        str(output_dir),
        "--quiet",
    ]
    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(project_root / "src")
        if not existing
        else os.pathsep.join([str(project_root / "src"), existing])
    )
    subprocess.check_call(cmd, env=env)
