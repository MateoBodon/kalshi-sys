from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from kalshi_alpha.exec.index_paper_ledger import LEDGER_ENV_KEY, log_index_paper_trade
from kalshi_alpha.exec.preflight_index import PreflightResult
from kalshi_alpha.exec.supervisor_index import SupervisorIndexConfig, _run_once, _run_window
from kalshi_alpha.sched import next_windows

ET = ZoneInfo("America/New_York")


class FakeWSListener:
    def __init__(self, fresh: bool = True) -> None:
        self.fresh = fresh
        self.started = False
        self.stopped = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def freshness(self, *, strict: bool, now: datetime | None = None) -> tuple[bool, float | None]:
        return (self.fresh, 10.0 if self.fresh else 5_000.0)


def _go_preflight(_: datetime) -> PreflightResult:
    return PreflightResult(go=True, reasons=[], details={})


def _no_go_preflight(_: datetime) -> PreflightResult:
    return PreflightResult(go=False, reasons=["env_missing"], details={})


@pytest.mark.asyncio
async def test_supervisor_skips_outside_window(tmp_path: Path) -> None:
    now = datetime(2025, 11, 3, 7, 0, tzinfo=ET)  # well before 10:00 ET hourly window
    config = SupervisorIndexConfig(now=now, offline=True, loop=False, quiet=True)
    calls: list[str] = []

    async def run() -> None:
        await _run_once(
            config,
            preflight_fn=_go_preflight,
            ws_factory=lambda: FakeWSListener(True),
            runner=lambda series, window, cfg, moment: calls.append(series),
        )

    await run()
    assert calls == []


@pytest.mark.asyncio
async def test_supervisor_runs_and_writes_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    now = datetime(2025, 11, 3, 9, 50, tzinfo=ET)  # inside hourly window (10:00 target)
    config = SupervisorIndexConfig(now=now, offline=True, loop=False, quiet=True)
    ledger_path = tmp_path / "index_paper.jsonl"
    monkeypatch.setenv(LEDGER_ENV_KEY, str(ledger_path))
    calls: list[str] = []

    def runner(series: str, window, cfg, moment: datetime) -> None:
        calls.append(series)
        log_index_paper_trade(
            {
                "series": series,
                "kalshi_market_id": f"{series}-MKT",
                "strike": 5000.0,
                "side": "yes",
                "price": 0.42,
                "size": 1,
                "ev_after_fees_cents": 5.0,
                "timestamp_et": moment,
            },
            ledger_path=ledger_path,
        )

    await _run_once(
        config,
        preflight_fn=_go_preflight,
        ws_factory=lambda: FakeWSListener(True),
        runner=runner,
    )

    assert sorted(calls) == ["INXU", "NASDAQ100U"]
    assert ledger_path.exists()
    lines = ledger_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


@pytest.mark.asyncio
async def test_supervisor_skips_on_preflight_failure(tmp_path: Path) -> None:
    now = datetime(2025, 11, 3, 9, 50, tzinfo=ET)
    config = SupervisorIndexConfig(now=now, offline=True, loop=False, quiet=True)
    calls: list[str] = []

    await _run_once(
        config,
        preflight_fn=_no_go_preflight,
        ws_factory=lambda: FakeWSListener(True),
        runner=lambda series, window, cfg, moment: calls.append(series),
    )

    assert calls == []


@pytest.mark.asyncio
async def test_transient_preflight_is_retried(tmp_path: Path) -> None:
    now = datetime(2025, 11, 3, 9, 50, tzinfo=ET)
    window = next_windows(now, limit=1)[0]
    config = SupervisorIndexConfig(now=now, offline=True, loop=False, quiet=True, preflight_retry_interval=30.0)
    ws = FakeWSListener(True)
    calls: list[str] = []

    ran, terminal = await _run_window(
        window,
        now_et=now,
        config=config,
        preflight_fn=lambda _: PreflightResult(go=False, reasons=["polygon_unreachable"], details={}),
        ws_listener=ws,
        runner=lambda series, w, cfg, moment: calls.append(series),
    )

    assert not ran
    assert not terminal
    assert calls == []

    ran2, terminal2 = await _run_window(
        window,
        now_et=now,
        config=config,
        preflight_fn=lambda _: PreflightResult(go=False, reasons=["calibration_missing:INX:noon"], details={}),
        ws_listener=ws,
        runner=lambda series, w, cfg, moment: calls.append(series),
    )

    assert not ran2
    assert terminal2
