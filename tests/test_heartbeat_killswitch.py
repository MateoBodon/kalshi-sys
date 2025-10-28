from __future__ import annotations

from argparse import Namespace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kalshi_alpha.exec.heartbeat import (
    heartbeat_stale,
    kill_switch_engaged,
    resolve_kill_switch_path,
    write_heartbeat,
)
from kalshi_alpha.exec.runners import scan_ladders
from kalshi_alpha.exec.runners.scan_ladders import Proposal, execute_broker
from kalshi_alpha.exec.state.orders import OutstandingOrdersState


def _sample_proposal() -> Proposal:
    return Proposal(
        market_id="M1",
        market_ticker="CPI-TEST",
        strike=270.0,
        side="YES",
        contracts=1,
        maker_ev=10.0,
        taker_ev=-9.0,
        maker_ev_per_contract=10.0,
        taker_ev_per_contract=-9.0,
        strategy_probability=0.55,
        market_yes_price=0.45,
        survival_market=0.52,
        survival_strategy=0.55,
        max_loss=100.0,
        strategy="CPI",
        metadata={},
    )


def test_write_heartbeat_and_staleness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from kalshi_alpha.datastore import paths as datastore_paths

    monkeypatch.setattr(datastore_paths, "PROC_ROOT", tmp_path / "data" / "proc")
    reference = datetime(2025, 1, 1, 15, 30, tzinfo=UTC)
    path = write_heartbeat(mode="scan:test", monitors={"lag": 1.2}, now=reference)
    assert path.exists()

    stale, payload = heartbeat_stale(threshold=timedelta(minutes=5), now=reference + timedelta(minutes=10))
    assert stale is True
    assert payload is not None and payload.get("mode") == "scan:test"

    fresh, _ = heartbeat_stale(threshold=timedelta(minutes=30), now=reference + timedelta(minutes=10))
    assert fresh is False


def test_kill_switch_engaged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from kalshi_alpha.datastore import paths as datastore_paths

    monkeypatch.setattr(datastore_paths, "PROC_ROOT", tmp_path / "data" / "proc")
    sentinel = resolve_kill_switch_path(None)
    assert not kill_switch_engaged(sentinel)
    sentinel.write_text("halt", encoding="utf-8")
    assert kill_switch_engaged(sentinel)


def test_execute_broker_halts_on_kill_switch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    isolated_data_roots: tuple[Path, Path],
) -> None:
    kill_switch_file = tmp_path / "kill_switch"
    kill_switch_file.write_text("halt", encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(scan_ladders, "_enforce_broker_guards", lambda *_, **__: None)

    args = Namespace(
        kill_switch_file=str(kill_switch_file),
        broker="dry",
        i_understand_the_risks=False,
    )

    with pytest.raises(RuntimeError, match="Kill switch engaged"):
        execute_broker(
            broker_mode="dry",
            proposals=[_sample_proposal()],
            args=args,
            monitors={},
            quiet=True,
            go_status=True,
        )

    state = OutstandingOrdersState.load()
    cancel_payload = state.cancel_all_request()
    assert cancel_payload is not None
    assert cancel_payload["reason"] == "kill_switch_engaged"
