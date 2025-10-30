from __future__ import annotations

from argparse import Namespace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from kalshi_alpha.core.kalshi_api import Event, Market, Orderbook, Series
from kalshi_alpha.exec.pipelines import daily
from kalshi_alpha.exec.runners import scan_ladders


def test_force_run_teny_shape_creates_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_data_roots: tuple[Path, Path],
) -> None:
    monkeypatch.chdir(tmp_path)

    class DummyClient:
        def __init__(self, *_args, **_kwargs) -> None:
            self._series = [Series(id="SER-TNEY", ticker="TNEY", name="TenY Close")]
            strikes = [3.5, 3.7, 3.9, 4.1]
            yes_prices = [0.42, 0.35, 0.28, 0.22]
            self._market = Market(
                id="MKT-TNEY",
                event_id="EV-TNEY",
                ticker="TNEY-CL",
                title="TenY Close",
                ladder_strikes=strikes,
                ladder_yes_prices=yes_prices,
            )
            self._event = Event(
                id="EV-TNEY",
                series_id="SER-TNEY",
                ticker="TNEY-CL",
                title="TenY Close",
            )
            self._orderbook = Orderbook(
                market_id="MKT-TNEY",
                bids=[{"price": 0.38, "size": 10}],
                asks=[{"price": 0.62, "size": 10}],
            )

        def get_series(self, *_, **__) -> list[Series]:
            return self._series

        def get_events(self, series_id: str, *_, **__) -> list[Event]:
            assert series_id == self._series[0].id
            return [self._event]

        def get_markets(self, event_id: str, *_, **__) -> list[Market]:
            assert event_id == self._event.id
            return [self._market]

        def get_orderbook(self, market_id: str, *_, **__) -> Orderbook:
            assert market_id == self._market.id
            return self._orderbook

    monkeypatch.setattr(daily, "KalshiPublicClient", DummyClient)

    config_src = Path(__file__).resolve().parents[1] / "configs" / "pal_policy.example.yaml"
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_dst = config_dir / "pal_policy.example.yaml"
    config_dst.write_text(config_src.read_text(encoding="utf-8"), encoding="utf-8")

    history_payload = [
        {
            "actual_close": 3.55,
            "prior_close": 3.48,
            "macro_shock": 0.02,
        },
        {
            "actual_close": 3.6,
            "prior_close": 3.55,
            "macro_shock": 0.03,
        },
    ]
    monkeypatch.setattr(scan_ladders, "_load_history", lambda *_: history_payload)

    monkeypatch.setattr(daily, "run_ingest", lambda *_, **__: None)

    def fake_calibrations(args, log, heartbeat_cb=None):
        if heartbeat_cb:
            heartbeat_cb("post_calibrate", {"calibration": "teny"})

    monkeypatch.setattr(daily, "run_calibrations", fake_calibrations)

    monkeypatch.setattr(
        daily,
        "run_quality_gates",
        lambda **_: daily.QualityGateResult(go=True, reasons=[], details={}),
    )
    monkeypatch.setattr(
        daily.drawdown,
        "check_limits",
        lambda *_, **__: SimpleNamespace(ok=True, reasons=[], metrics={}),
    )
    monkeypatch.setattr(daily, "load_quality_gate_config", lambda *_: SimpleNamespace())
    monkeypatch.setattr(daily, "resolve_quality_gate_config_path", lambda: Path("dummy"))

    def fake_archive(**_kwargs):
        artifacts_dir = Path("reports/_artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        replay_path = artifacts_dir / "replay_ev.parquet"
        if not replay_path.exists():
            pl.DataFrame({"market_id": ["MKT-TNEY"], "ev": [0.0]}).write_parquet(replay_path)
        return None

    monkeypatch.setattr(daily, "_archive_and_replay", fake_archive)

    args = Namespace(
        offline=True,
        online=False,
        driver_fixtures=str(tmp_path / "fixtures"),
        scanner_fixtures=str(tmp_path / "fixtures"),
        kelly_cap=0.15,
        fill_alpha="auto",
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

    daily.run_mode("teny_close", args)

    report_dir = Path("reports") / "TNEY"
    report_files = list(report_dir.glob("*.md"))
    assert report_files, "Expected TenY report to be rendered"

    artifacts_dir = Path("reports/_artifacts")
    cdf_path = artifacts_dir / "cdf_diffs.parquet"
    replay_path = artifacts_dir / "replay_ev.parquet"
    assert cdf_path.exists(), "Expected cdf_diffs.parquet to be written"
    assert replay_path.exists(), "Expected replay_ev.parquet to be written"

    contents = report_files[0].read_text(encoding="utf-8")
    assert "FORCE-RUN (DRY)" in contents
