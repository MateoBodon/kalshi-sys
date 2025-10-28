from __future__ import annotations

from pathlib import Path

from kalshi_alpha.brokers.kalshi.base import BrokerOrder
from kalshi_alpha.exec.state.orders import OutstandingOrdersState


def _sample_order(key: str) -> BrokerOrder:
    return BrokerOrder(
        idempotency_key=key,
        market_id="MKT-123",
        strike=250.0,
        side="YES",
        price=0.45,
        contracts=5,
        probability=0.6,
        metadata={"note": "demo"},
    )


def test_state_records_and_persists_orders(tmp_path: Path) -> None:
    state_path = tmp_path / "orders.json"
    state = OutstandingOrdersState.load(state_path)
    assert state.summary() == {"dry": 0, "live": 0}

    order = _sample_order("abc")
    state.record_submission("live", [order])
    assert state.summary()["live"] == 1
    assert state.total() == 1

    reloaded = OutstandingOrdersState.load(state_path)
    assert reloaded.summary()["live"] == 1
    stored = reloaded.outstanding_for("live")
    assert "abc" in stored
    assert stored["abc"]["market_id"] == "MKT-123"


def test_state_mark_cancel_all(tmp_path: Path) -> None:
    state = OutstandingOrdersState.load(tmp_path / "orders.json")
    state.mark_cancel_all("no_go", modes=["live"])
    payload = state.cancel_all_request()
    assert payload is not None
    assert payload["reason"] == "no_go"
    assert payload["modes"] == ["live"]


def test_state_reconcile_removes_missing(tmp_path: Path) -> None:
    state_path = tmp_path / "orders.json"
    state = OutstandingOrdersState.load(state_path)
    order_a = _sample_order("order-a")
    order_b = _sample_order("order-b")
    state.record_submission("live", [order_a, order_b])
    removed = state.reconcile("live", ["order-a"])
    assert removed == ["order-b"]
    reloaded = OutstandingOrdersState.load(state_path)
    assert set(reloaded.outstanding_for("live")) == {"order-a"}
