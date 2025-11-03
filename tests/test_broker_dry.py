from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from kalshi_alpha.brokers.kalshi.base import BrokerOrder
from kalshi_alpha.brokers.kalshi.dry import DryBroker
from kalshi_alpha.exec.runners.scan_ladders import Proposal, execute_broker


def test_dry_broker_serializes_and_audits(tmp_path: Path) -> None:
    artifacts = tmp_path / "reports" / "_artifacts"
    audit_dir = tmp_path / "data" / "proc" / "audit"
    broker = DryBroker(artifacts_dir=artifacts, audit_dir=audit_dir)

    order = BrokerOrder(
        idempotency_key="abc",
        market_id="M1",
        strike=270.0,
        side="YES",
        price=0.45,
        contracts=10,
        probability=0.52,
    )
    duplicate = BrokerOrder(
        idempotency_key="abc",
        market_id="M1",
        strike=270.0,
        side="YES",
        price=0.45,
        contracts=10,
        probability=0.52,
    )

    broker.place([order, duplicate])

    orders = sorted(artifacts.glob("orders_*.json"))
    assert len(orders) == 1
    payload = json.loads(orders[0].read_text())
    assert len(payload) == 1
    assert payload[0]["market_id"] == "M1"

    audit_files = sorted(audit_dir.glob("orders_*.jsonl"))
    assert len(audit_files) == 1
    lines = [json.loads(line) for line in audit_files[0].read_text().splitlines()]
    actions = [entry["action"] for entry in lines]
    assert actions.count("place") == 1
    assert actions.count("duplicate") == 1


def test_execute_broker_refuses_on_no_go(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _seed_policy(tmp_path)
    proposal = _sample_proposal()
    args = Namespace(
        pal_policy=None,
        max_loss_per_strike=None,
        portfolio_config=None,
        max_var=None,
        daily_loss_cap=None,
        weekly_loss_cap=None,
        broker="dry",
    )
    with pytest.raises(RuntimeError, match="NO-GO"):
        execute_broker(
            broker_mode="dry",
            proposals=[proposal],
            args=args,
            monitors={},
            quiet=True,
            go_status=False,
        )


def test_execute_broker_writes_orders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _seed_policy(tmp_path)
    proposal = _sample_proposal()
    args = Namespace(
        pal_policy=None,
        max_loss_per_strike=None,
        portfolio_config=None,
        max_var=None,
        daily_loss_cap=10_000.0,
        weekly_loss_cap=50_000.0,
        broker="dry",
    )

    status = execute_broker(
        broker_mode="dry",
        proposals=[proposal],
        args=args,
        monitors={"non_monotone_ladders": 0},
        quiet=True,
        go_status=True,
    )

    assert status is not None
    assert status.get("orders_recorded") == 1

    orders = sorted(Path("reports/_artifacts").glob("orders_*.json"))
    assert len(orders) == 1

    audit_files = sorted(Path("data/proc/audit").glob("orders_*.jsonl"))
    assert len(audit_files) == 1


def _seed_policy(root: Path) -> None:
    configs_dir = root / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.joinpath("pal_policy.yaml").write_text(
        """
series: CPI
default_max_loss: 5000
per_strike:
  "CPI-TEST": 2500
""",
        encoding="utf-8",
    )


def _sample_proposal() -> Proposal:
    return Proposal(
        market_id="M1",
        market_ticker="CPI-TEST",
        strike=270.0,
        side="YES",
        contracts=5,
        maker_ev=150.0,
        taker_ev=-200.0,
        maker_ev_per_contract=30.0,
        taker_ev_per_contract=-40.0,
        strategy_probability=0.55,
        market_yes_price=0.45,
        survival_market=0.52,
        survival_strategy=0.55,
        max_loss=1125.0,
        strategy="CPI",
        series="CPI",
        metadata={},
    )
