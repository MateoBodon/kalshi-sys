from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kalshi_alpha.core.pricing import Liquidity
from kalshi_alpha.exec.ledger import FillRecord, PaperLedger
from kalshi_alpha.exec.pipelines import daily
from kalshi_alpha.exec.runners.scan_ladders import Proposal


class _DummyOutcome:
    def __init__(self, proposals: list[Proposal]) -> None:
        self.proposals = proposals
        self.monitors: dict[str, object] = {}
        self.cdf_diffs: list[dict[str, object]] = []
        self.mispricings: list[dict[str, object]] = []
        self.events: list[object] = []
        self.markets: list[object] = []
        self.model_metadata: dict[str, object] = {}
        self.series = SimpleNamespace(ticker="TNEY", id="SER-TNEY")


def _proposal() -> Proposal:
    return Proposal(
        market_id="MKT",
        market_ticker="TNEY-TEST",
        strike=4.2,
        side="YES",
        contracts=10,
        maker_ev=2.0,
        taker_ev=0.0,
        maker_ev_per_contract=0.2,
        taker_ev_per_contract=0.0,
        strategy_probability=0.6,
        market_yes_price=0.4,
        survival_market=0.3,
        survival_strategy=0.6,
        max_loss=5.0,
        strategy="TNEY",
        metadata={},
    )


def test_fill_realism_gate_triggers_no_go(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    from kalshi_alpha.datastore import paths as datastore_paths

    raw_root = tmp_path / "data" / "raw"
    proc_root = tmp_path / "data" / "proc"
    raw_root.mkdir(parents=True, exist_ok=True)
    proc_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(datastore_paths, "RAW_ROOT", raw_root)
    monkeypatch.setattr(datastore_paths, "PROC_ROOT", proc_root)
    monkeypatch.setattr(daily, "PROC_ROOT", proc_root)
    monkeypatch.setattr(daily, "RAW_ROOT", raw_root)

    proposal = _proposal()

    outcome = _DummyOutcome([proposal])
    monkeypatch.setattr(daily, "scan_series", lambda *_, **__: outcome)

    ledger = PaperLedger()
    ledger.set_series("TNEY")
    ledger.record(
        FillRecord(
            proposal=proposal,
            fill_price=0.4,
            expected_value=2.0,
            liquidity=Liquidity.MAKER,
            slippage=0.0,
            expected_contracts=10,
            expected_fills=10,
            fill_ratio=0.95,
            slippage_mode="top",
            impact_cap=0.0,
            fees_maker=0.0,
            pnl_simulated=2.0,
            alpha_row=0.6,
            size_throttled=False,
        )
    )

    monkeypatch.setattr(daily, "simulate_fills", lambda *_, **__: ledger)
    monkeypatch.setattr(daily, "_compute_exposure_summary", lambda *_: {})
    monkeypatch.setattr(daily, "_write_cdf_diffs", lambda *_: None)
    monkeypatch.setattr(daily, "_archive_and_replay", lambda **_: None)
    monkeypatch.setattr(daily, "write_proposals", lambda **_: tmp_path / "proposals.json")
    monkeypatch.setattr(daily, "run_ingest", lambda *_, **__: None)
    monkeypatch.setattr(daily, "run_calibrations", lambda *_, **__: None)
    monkeypatch.setattr(daily, "load_quality_gate_config", lambda *_: SimpleNamespace())
    monkeypatch.setattr(daily, "resolve_quality_gate_config_path", lambda: Path("dummy"))
    monkeypatch.setattr(
        daily,
        "run_quality_gates",
        lambda **_: daily.QualityGateResult(go=True, reasons=[], details={}),
    )
    monkeypatch.setattr(
        daily.PALPolicy,
        "from_yaml",
        classmethod(lambda cls, path: cls(series="TNEY", default_max_loss=1000.0)),
    )

    args = Namespace(
        offline=True,
        online=False,
        driver_fixtures=str(tmp_path / "fixtures"),
        scanner_fixtures=str(tmp_path / "fixtures"),
        kelly_cap=0.15,
        fill_alpha="0.6",
        slippage_mode="top",
        impact_cap=0.02,
        report=True,
        paper_ledger=True,
        broker="dry",
        allow_no_go=False,
        mispricing_only=False,
        max_legs=4,
        prob_sum_gap_threshold=0.0,
        model_version="v15",
        kill_switch_file=None,
        when=None,
        daily_loss_cap=None,
        weekly_loss_cap=None,
        force_refresh=False,
        paper=False,
        force_run=True,
        window_et=None,
    )

    fixtures_root = Path(args.driver_fixtures)
    (fixtures_root / "kalshi").mkdir(parents=True, exist_ok=True)

    config_src = Path(__file__).resolve().parents[1] / "configs" / "pal_policy.example.yaml"
    config_dst_dir = tmp_path / "configs"
    config_dst_dir.mkdir(parents=True, exist_ok=True)
    config_dst = config_dst_dir / "pal_policy.example.yaml"
    if config_src.exists():
        config_dst.write_text(config_src.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        config_dst.write_text("series: TNEY\n", encoding="utf-8")

    daily.run_mode("teny_close", args)

    go_file = Path("reports/_artifacts/go_no_go.json")
    payload = json.loads(go_file.read_text(encoding="utf-8"))
    assert payload["go"] is False
    assert "fill_realism_miss" in payload["reasons"]
