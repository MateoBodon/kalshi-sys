from argparse import Namespace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from kalshi_alpha.exec.pilot import apply_pilot_mode, load_pilot_config


def _write_pilot_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "pilot.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_load_pilot_config_from_yaml(tmp_path: Path) -> None:
    config_path = _write_pilot_config(
        tmp_path,
        """
        pilot:
          session_prefix: test
          allowed_series:
            - CPI
          max_contracts_per_order: 2
          max_unique_bins: 2
          require_live_broker: true
          enforce_maker_only: true
          require_acknowledgement: true
          max_daily_loss: 75
          max_weekly_loss: 125
        """
        .strip()
        + "\n",
    )
    config = load_pilot_config(config_path)
    assert config.session_prefix == "test"
    assert config.allowed_series == ("CPI",)
    assert config.max_contracts_per_order == 2
    assert config.max_unique_bins == 2
    assert config.require_live_broker is True
    assert config.enforce_maker_only is True
    assert config.max_daily_loss == pytest.approx(75)
    assert config.max_weekly_loss == pytest.approx(125)


def test_apply_pilot_mode_enforces_constraints(tmp_path: Path) -> None:
    config_path = _write_pilot_config(
        tmp_path,
        """
        pilot:
          session_prefix: test
          allowed_series:
            - CPI
          max_contracts_per_order: 2
          max_unique_bins: 2
          require_live_broker: true
          enforce_maker_only: true
          require_acknowledgement: true
          max_daily_loss: 90
          max_weekly_loss: 150
        """
        .strip()
        + "\n",
    )
    args = Namespace(
        pilot=True,
        pilot_config=config_path,
        series="CPI",
        contracts=5,
        maker_only=False,
        broker="live",
        i_understand_the_risks=True,
        offline=False,
        online=False,
        sizing="kelly",
        max_legs=6,
        daily_loss_cap=None,
        weekly_loss_cap=200.0,
    )

    session = apply_pilot_mode(
        args,
        now=datetime(2025, 11, 2, 12, 0, tzinfo=UTC),
        token_factory=lambda _: "abc123",
    )

    assert session is not None
    assert session.session_id == "test-cpi-20251102T120000Z-abc123"
    assert args.maker_only is True
    assert args.contracts == 2
    assert args.sizing == "fixed"
    assert args.max_legs == 2
    assert args.online is True
    assert args.daily_loss_cap == pytest.approx(90.0)
    assert args.weekly_loss_cap == pytest.approx(150.0)
    assert args.pilot_session_id == session.session_id
    assert args.pilot_max_contracts == 2
    assert args.pilot_max_unique_bins == 2


def test_apply_pilot_mode_blocks_disallowed_series(tmp_path: Path) -> None:
    config_path = _write_pilot_config(
        tmp_path,
        """
        pilot:
          allowed_series:
            - CPI
        """
        .strip()
        + "\n",
    )
    args = Namespace(
        pilot=True,
        pilot_config=config_path,
        series="TNEY",
        contracts=1,
        maker_only=True,
        broker="live",
        i_understand_the_risks=True,
        offline=False,
        online=True,
        sizing="fixed",
        max_legs=2,
        daily_loss_cap=None,
        weekly_loss_cap=None,
    )

    with pytest.raises(ValueError):
        apply_pilot_mode(args, now=datetime(2025, 11, 2, tzinfo=UTC))
