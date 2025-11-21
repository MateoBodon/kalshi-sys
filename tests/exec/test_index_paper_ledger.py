import json
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from kalshi_alpha.exec.index_paper_ledger import LEDGER_ENV_KEY, log_index_paper_trade
from kalshi_alpha.exec.runners import scan_ladders


def test_log_index_paper_trade_writes_jsonl(tmp_path):
    ledger_path = tmp_path / "index_paper.jsonl"
    timestamp = datetime(2025, 11, 20, 12, 0, tzinfo=ZoneInfo("America/New_York"))
    record = {
        "timestamp_et": timestamp,
        "series": "inx",
        "window": "hourly-1200",
        "kalshi_market_id": "mkt-123",
        "market_ticker": "INXU-H1200",
        "strike": 5440.0,
        "side": "YES",
        "price": 0.42,
        "size": 1,
        "fill_prob": 0.65,
        "ev_after_fees_cents": 4.2,
        "mode": "ignored",
    }

    path = log_index_paper_trade(record, ledger_path=ledger_path)

    assert path == ledger_path
    contents = ledger_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(contents) == 1
    payload = json.loads(contents[0])
    assert payload["series"] == "INX"
    assert payload["side"] == "buy"  # normalized
    assert payload["mode"] == "dry"
    assert payload["window"] == "hourly-1200"
    assert payload["ev_after_fees_cents"] == pytest.approx(4.2)
    assert payload["timestamp_et"].startswith("2025-11-20T12:00:00")


def test_execute_broker_logs_index_trade(monkeypatch, tmp_path):
    ledger_path = tmp_path / "index_paper.jsonl"
    monkeypatch.setenv(LEDGER_ENV_KEY, str(ledger_path))

    # Stub guardrail helpers to keep the test hermetic.
    monkeypatch.setattr(
        scan_ladders,
        "_quality_gate_for_broker",
        lambda *_, **__: scan_ladders.QualityGateResult(go=True, reasons=[], details={}),
    )
    monkeypatch.setattr(scan_ladders, "_enforce_broker_guards", lambda *_, **__: None)

    class DummyState:
        @classmethod
        def load(cls):
            return cls()

        def mark_cancel_all(self, *_, **__):
            return None

        def record_submission(self, *_, **__):
            return None

    monkeypatch.setattr(scan_ladders, "OutstandingOrdersState", DummyState)
    monkeypatch.setattr(
        scan_ladders,
        "resolve_kill_switch_path",
        lambda *_, **__: tmp_path / "ks",
    )

    class DummyBroker:
        mode = "dry"

        def __init__(self):
            self.orders = []

        def place(self, orders):
            self.orders.extend(orders)

        def status(self):
            return {"mode": "dry", "orders_recorded": len(self.orders)}

    monkeypatch.setattr(scan_ladders, "create_broker", lambda *_, **__: DummyBroker())

    proposal = scan_ladders.Proposal(
        market_id="mkt-abc",
        market_ticker="INXU-H1500",
        strike=5500.0,
        side="YES",
        contracts=1,
        maker_ev=0.07,
        taker_ev=0.0,
        maker_ev_per_contract=0.07,
        taker_ev_per_contract=0.0,
        strategy_probability=0.55,
        market_yes_price=0.42,
        survival_market=0.5,
        survival_strategy=0.55,
        max_loss=0.58,
        strategy="index",
        series="INXU",
        metadata={},
    )
    monitors = {"scheduler_window": {"label": "hourly-1500"}}
    args = SimpleNamespace(series="INXU", kill_switch_file=None, i_understand_the_risks=True)

    scan_ladders.execute_broker(
        broker_mode="dry",
        proposals=[proposal],
        args=args,
        monitors=monitors,
        quiet=True,
        go_status=True,
    )

    contents = ledger_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(contents) == 1
    payload = json.loads(contents[0])
    assert payload["series"] == "INXU"
    assert payload["window"] == "hourly-1500"
    assert payload["side"] == "buy"
    assert payload["ev_after_fees_cents"] == pytest.approx(7.0)
    assert payload["kalshi_market_id"] == "mkt-abc"
