from datetime import date
from pathlib import Path

from kalshi_alpha.exec import scoreboard_index_paper as scoreboard


def test_load_and_aggregate_sample_ledger():
    ledger_path = Path("tests/data_fixtures/ledger_index_paper/sample.jsonl")
    start_date = date(2025, 11, 18)
    end_date = date(2025, 11, 19)

    entries = scoreboard.load_entries(ledger_path, start_date=start_date, end_date=end_date)
    assert len(entries) == 3
    summaries = scoreboard.aggregate(entries)
    keyed = {(
        row["date"],
        row["series"],
        row["window"],
    ): row for row in summaries}

    assert keyed[(date(2025, 11, 18), "INXU", "hourly-1300")]["trades"] == 1
    assert keyed[(date(2025, 11, 18), "INX", "close-1600")]["ev_sum_cents"] == 8.5
    assert keyed[(date(2025, 11, 19), "NASDAQ100U", "hourly-1100")]["avg_ev_cents"] == 6.2


def test_scoreboard_cli_writes_markdown(tmp_path):
    ledger_path = Path("tests/data_fixtures/ledger_index_paper/sample.jsonl")
    output_path = tmp_path / "score.md"
    rc = scoreboard.main(
        [
            "--start-date",
            "2025-11-18",
            "--end-date",
            "2025-11-19",
            "--ledger",
            str(ledger_path),
            "--output",
            str(output_path),
        ]
    )
    assert rc == 0
    assert output_path.exists()
    contents = output_path.read_text(encoding="utf-8")
    assert "Index Paper Scoreboard" in contents
    assert "INX" in contents
