from __future__ import annotations

from pathlib import Path

import pytest

from kalshi_alpha.brokers.kalshi.base import BrokerOrder
from kalshi_alpha.exec.runners import orders_doctor
from kalshi_alpha.exec.state.orders import OutstandingOrdersState


def _sample_order(key: str, market: str, side: str = "YES") -> BrokerOrder:
    return BrokerOrder(
        idempotency_key=key,
        market_id=market,
        strike=270.0,
        side=side,
        price=0.45,
        contracts=5,
        probability=0.55,
        metadata={"note": "test"},
    )


def test_orders_doctor_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from kalshi_alpha.datastore import paths as datastore_paths

    monkeypatch.chdir(tmp_path)
    proc_root = tmp_path / "data" / "proc"
    monkeypatch.setattr(datastore_paths, "PROC_ROOT", proc_root)

    state = OutstandingOrdersState.load()
    state.record_submission("dry", [_sample_order("dry-1", "M1")])
    state.record_submission("dry", [_sample_order("dry-2", "M2", side="NO")])
    state.record_submission("live", [_sample_order("live-1", "LIVE")])

    state_path = proc_root / "state" / "orders.json"

    orders_doctor.main(["--show", "--state-path", str(state_path)])
    output = capsys.readouterr().out
    assert "Outstanding orders: 3" in output
    assert "- dry: 2" in output
    assert "- live: 1" in output
    assert "DRY orders:" in output
    assert "dry-1" in output

    orders_doctor.main(["--reconcile", "--state-path", str(state_path)])
    capsys.readouterr()
    state_after_reconcile = OutstandingOrdersState.load(state_path)
    dry_bucket = state_after_reconcile.outstanding_for("dry")
    assert all(record["status"] == "cancelled" for record in dry_bucket.values())
    cancel_payload = state_after_reconcile.cancel_all_request()
    assert cancel_payload is not None and cancel_payload["reason"] == "orders_doctor_reconcile"

    orders_doctor.main(["--clear-dry", "--show", "--state-path", str(state_path)])
    output = capsys.readouterr().out
    assert "Outstanding orders: 1" in output
    assert "- dry: 0" in output
    assert "- live: 1" in output
    final_state = OutstandingOrdersState.load(state_path)
    assert final_state.outstanding_for("dry") == {}
    assert final_state.cancel_all_request() is None
