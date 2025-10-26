from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def fixtures_root() -> Path:
    return Path(__file__).parent / "data_fixtures"


@pytest.fixture
def offline_fixtures_root() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def pal_policy_path(tmp_path: Path) -> Path:
    policy_src = Path("configs/pal_policy.example.yaml")
    policy_copy = tmp_path / "pal_policy.yaml"
    policy_copy.write_text(policy_src.read_text(encoding="utf-8"), encoding="utf-8")
    return policy_copy
