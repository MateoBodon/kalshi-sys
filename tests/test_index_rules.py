from __future__ import annotations

import pytest

from kalshi_alpha.config import load_index_rulebook, lookup_index_rule


def test_index_rulebook_loader_parses_series() -> None:
    rulebook = load_index_rulebook()
    assert pytest.approx(rulebook.tick_size_usd, rel=1e-9) == 0.01
    assert rulebook.position_limit_usd == 7_000_000
    assert {"INX", "INXU", "NASDAQ100", "NASDAQ100U"}.issubset(set(rulebook.series))


def test_lookup_index_rule_returns_expected_fields() -> None:
    inx_rule = lookup_index_rule("INX")
    assert inx_rule.series == "INX"
    assert "16:00" in inx_rule.evaluation_time_et
    assert "closing" in inx_rule.evaluation_clause.lower()
    assert inx_rule.tick_size_usd == pytest.approx(0.01)
    assert inx_rule.position_limit_usd == 7_000_000

