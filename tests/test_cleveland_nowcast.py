from __future__ import annotations

from pathlib import Path

import pytest

from kalshi_alpha.drivers import cleveland_nowcast


def test_cleveland_nowcast_offline(offline_fixtures_root: Path) -> None:
    series = cleveland_nowcast.fetch_nowcast(offline=True, fixtures_dir=offline_fixtures_root / "cleveland_nowcast")
    assert "headline" in series and "core" in series


def test_cleveland_nowcast_online_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    fixtures = Path("tests/fixtures/cleveland_nowcast")
    page_bytes = fixtures.joinpath("sample_page.html").read_bytes()
    json_bytes = fixtures.joinpath("sample_nowcast.json").read_bytes()
    responses = [page_bytes, json_bytes]

    def fake_fetch(url: str, cache_path: Path, session=None, force_refresh=False):  # type: ignore[override]
        return responses.pop(0)

    monkeypatch.setattr(cleveland_nowcast, "fetch_with_cache", fake_fetch)
    series = cleveland_nowcast.fetch_nowcast(offline=False, fixtures_dir=None)
    assert pytest.approx(series["headline"].value, abs=1e-6) == 0.22
    assert pytest.approx(series["core"].value, abs=1e-6) == 0.18
