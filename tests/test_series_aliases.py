from __future__ import annotations

from kalshi_alpha.utils.series import (
    index_series_query_candidates,
    normalize_index_series,
    normalize_index_series_list,
)


def test_normalize_index_series_aliases() -> None:
    assert normalize_index_series("SPX") == "INX"
    assert normalize_index_series("spxu") == "INXU"
    assert normalize_index_series("NDX") == "NASDAQ100"
    assert normalize_index_series("I:NDXU") == "NASDAQ100U"
    assert normalize_index_series("KXNDX") == "NASDAQ100"


def test_normalize_index_series_list_dedupes() -> None:
    result = normalize_index_series_list(["SPX", "INXU", "SPX", "NDX"])
    assert result == ("INX", "INXU", "NASDAQ100")


def test_index_series_query_candidates() -> None:
    assert index_series_query_candidates("SPXU") == ("SPXU", "INXU", "KXINXU")
    assert index_series_query_candidates("KXINX") == ("KXINX", "INX")
