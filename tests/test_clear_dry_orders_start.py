from __future__ import annotations

from pathlib import Path

from kalshi_alpha.brokers.kalshi.base import BrokerOrder
from kalshi_alpha.exec.runners import scan_ladders
from kalshi_alpha.exec.state.orders import OutstandingOrdersState


def test_clear_dry_orders_start_removes_orders(tmp_path: Path, capsys) -> None:
    state_path = tmp_path / "orders.json"
    state = OutstandingOrdersState.load(state_path)

    orders = [
        BrokerOrder(
            idempotency_key=f"dry-{idx}",
            market_id=f"M{idx}",
            strike=100 + idx,
            side="YES",
            price=0.45,
            contracts=5,
            probability=0.55,
        )
        for idx in range(2)
    ]
    state.record_submission("dry", orders)
    assert state.summary()["dry"] == 2

    summary = scan_ladders._clear_dry_orders_start(
        enabled=True,
        broker_mode="dry",
        quiet=False,
        state=state,
    )

    captured = capsys.readouterr().out
    assert "Cleared dry orders" in captured
    assert "Outstanding orders: 0" in captured
    assert summary["dry"] == 0
    assert OutstandingOrdersState.load(state_path).summary()["dry"] == 0
