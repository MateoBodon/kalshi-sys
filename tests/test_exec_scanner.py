from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from kalshi_alpha.core.kalshi_api import KalshiPublicClient
from kalshi_alpha.core.risk import PALGuard, PALPolicy, PortfolioConfig, PortfolioRiskManager
from kalshi_alpha.drivers.aaa_gas import fetch as aaa_fetch
from kalshi_alpha.exec.ledger import simulate_fills
from kalshi_alpha.exec.reports import write_markdown_report
from kalshi_alpha.exec.runners.scan_ladders import (
    BinConstraintEntry,
    BinConstraintResolver,
    _attach_series_metadata,
    _strategy_pmf_for_series,
    scan_series,
    write_proposals,
)
from kalshi_alpha.exec.scanners import cpi
from kalshi_alpha.exec.scanners.utils import expected_value_summary


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
    pmf, metadata = _strategy_pmf_for_series(
        series="CLAIMS",
        strikes=strikes,
        fixtures_dir=offline_fixtures_root,
        override="auto",
        offline=True,
    )
    assert metadata.get("model_version") == "v15"
    assert len(pmf) >= 1
    assert abs(sum(entry.probability for entry in pmf) - 1.0) < 1e-6


def test_strategy_router_teny_pmf(offline_fixtures_root: Path) -> None:
    strikes = [4.2, 4.3, 4.4]
    pmf, metadata = _strategy_pmf_for_series(
        series="TNEY",
        strikes=strikes,
        fixtures_dir=offline_fixtures_root,
        override="auto",
        offline=True,
    )
    assert metadata.get("model_version") == "v15"
    assert len(pmf) >= len(strikes)
    assert abs(sum(entry.probability for entry in pmf) - 1.0) < 1e-6


def test_teny_ev_shrink_applies_to_proposals(
    fixtures_root: Path, offline_fixtures_root: Path
) -> None:
    client = KalshiPublicClient(offline_dir=fixtures_root / "kalshi", use_offline=True)
    policy = PALPolicy(series="TNEY", default_max_loss=10_000.0)
    guard = PALGuard(policy)

    outcome = scan_series(
        series="TNEY",
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
        ev_honesty_shrink=0.5,
    )
    proposals = outcome.proposals
    assert proposals
    assert outcome.monitors.get("ev_honesty_shrink") == pytest.approx(0.5)

    for proposal in proposals:
        metadata = proposal.metadata or {}
        assert metadata.get("ev_shrink") == pytest.approx(0.5)
        ev_values = expected_value_summary(
            contracts=1,
            yes_price=proposal.market_yes_price,
            event_probability=proposal.strategy_probability,
            series=proposal.series,
            market_name=proposal.market_ticker,
        )
        key = "maker_yes" if proposal.side.upper() == "YES" else "maker_no"
        raw_ev_per_contract = ev_values[key]
        assert proposal.maker_ev_per_contract == pytest.approx(raw_ev_per_contract, rel=1e-6)
        shrunk = metadata.get("ev_shrunk")
        assert isinstance(shrunk, dict)
        assert shrunk.get("maker_per_contract") == pytest.approx(raw_ev_per_contract * 0.5, rel=1e-6)
        total_values = expected_value_summary(
            contracts=proposal.contracts,
            yes_price=proposal.market_yes_price,
            event_probability=proposal.strategy_probability,
            series=proposal.series,
            market_name=proposal.market_ticker,
        )
        raw_total = total_values[key]
        assert proposal.maker_ev == pytest.approx(raw_total, rel=1e-6)
        assert shrunk.get("maker_total") == pytest.approx(raw_total * 0.5, rel=1e-6)


def test_strategy_router_weather_pmf(offline_fixtures_root: Path) -> None:
    strikes = [55, 60, 65]
    pmf, metadata = _strategy_pmf_for_series(
        series="WEATHER",
        strikes=strikes,
        fixtures_dir=offline_fixtures_root,
        override="auto",
        offline=True,
    )
    assert metadata.get("model_version") == "v15"
    assert len(pmf) >= len(strikes)
    assert abs(sum(entry.probability for entry in pmf) - 1.0) < 1e-6


def test_bin_constraints_scale_contracts(fixtures_root: Path, offline_fixtures_root: Path) -> None:
    client = KalshiPublicClient(offline_dir=fixtures_root / "kalshi", use_offline=True)
    policy = PALPolicy(series="CPI", default_max_loss=10_000.0)
    base_guard = PALGuard(policy)
    base_outcome = scan_series(
        series="CPI",
        client=client,
        min_ev=0.01,
        contracts=5,
        pal_guard=base_guard,
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
    assert base_outcome.proposals
    reference = base_outcome.proposals[0]

    weight = 0.4
    expected_contracts = max(0, int(math.floor(reference.contracts * weight + 1e-9)))
    assert expected_contracts > 0, "weight should not zero out contract for this test"

    resolver = BinConstraintResolver(
        [
            BinConstraintEntry(
                series="CPI",
                market_id=reference.market_id,
                market_ticker=reference.market_ticker,
                strike=reference.strike,
                side=reference.side,
                weight=weight,
                cap=None,
                sources=("auto_ev_honesty",),
            )
        ]
    )

    client_adjusted = KalshiPublicClient(offline_dir=fixtures_root / "kalshi", use_offline=True)
    guard_adjusted = PALGuard(policy)
    adjusted_outcome = scan_series(
        series="CPI",
        client=client_adjusted,
        min_ev=0.01,
        contracts=5,
        pal_guard=guard_adjusted,
        driver_fixtures=offline_fixtures_root,
        strategy_name="auto",
        maker_only=True,
        allow_tails=False,
        risk_manager=None,
        max_var=None,
        offline=True,
        sizing_mode="fixed",
        kelly_cap=0.25,
        bin_constraints=resolver,
    )

    matches = [
        proposal
        for proposal in adjusted_outcome.proposals
        if proposal.market_id == reference.market_id
        and proposal.strike == reference.strike
        and proposal.side == reference.side
    ]
    assert matches
    adjusted = matches[0]
    assert adjusted.contracts == expected_contracts
    metadata = adjusted.metadata or {}
    assert "bin_constraint" in metadata
    summary = adjusted_outcome.monitors.get("ev_honesty_constraints")
    assert isinstance(summary, dict)
    assert summary.get("applied", 0) >= 1
    assert summary.get("dropped", 0) == 0


def test_bin_constraints_drop_bins(fixtures_root: Path, offline_fixtures_root: Path) -> None:
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
    assert outcome.proposals
    target = outcome.proposals[0]

    resolver = BinConstraintResolver(
        [
            BinConstraintEntry(
                series="CPI",
                market_id=target.market_id,
                market_ticker=target.market_ticker,
                strike=target.strike,
                side=target.side,
                weight=0.0,
                cap=0,
                sources=("manual_override",),
            )
        ]
    )

    client_drop = KalshiPublicClient(offline_dir=fixtures_root / "kalshi", use_offline=True)
    guard_drop = PALGuard(policy)
    dropped_outcome = scan_series(
        series="CPI",
        client=client_drop,
        min_ev=0.01,
        contracts=5,
        pal_guard=guard_drop,
        driver_fixtures=offline_fixtures_root,
        strategy_name="auto",
        maker_only=True,
        allow_tails=False,
        risk_manager=None,
        max_var=None,
        offline=True,
        sizing_mode="fixed",
        kelly_cap=0.25,
        bin_constraints=resolver,
    )

    assert all(
        not (
            proposal.market_id == target.market_id
            and proposal.side == target.side
            and proposal.strike == target.strike
        )
        for proposal in dropped_outcome.proposals
    )
    summary = dropped_outcome.monitors.get("ev_honesty_constraints")
    assert isinstance(summary, dict)
    assert summary.get("dropped", 0) >= 1
    assert resolver.summary().get("dropped", 0) >= 1


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
    subprocess.check_call(cmd, env=env)  # noqa: S603
