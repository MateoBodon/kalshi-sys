from pathlib import Path

from scripts.build_index_panel_polygon import build_panel
from kalshi_alpha.drivers.index_polygon import load_minutes


FIXTURE_ROOT = Path("tests/data_fixtures/index_panel_fast/raw/polygon/index")


def test_builds_panel_with_features(tmp_path):
    output_path = tmp_path / "panel.parquet"
    panel = build_panel(
        symbols=("I:SPX", "I:NDX"),
        input_root=FIXTURE_ROOT,
        output_path=output_path,
    )
    assert output_path.exists()
    assert panel.height > 0
    expected_cols = {
        "timestamp",
        "timestamp_et",
        "symbol",
        "price",
        "minutes_to_noon",
        "minutes_to_close",
        "day_of_week",
        "realized_vol_30m",
    }
    assert expected_cols.issubset(set(panel.columns))
    # Ensure ET timestamps carry timezone information.
    assert panel.schema["timestamp_et"].time_zone == "America/New_York"


def test_loader_respects_date_filters():
    start = "2025-10-31"
    df = load_minutes(
        ("I:SPX",),
        start_date=start,
        end_date=start,
        base_root=FIXTURE_ROOT,
    )
    assert not df.is_empty()
    days = df.get_column("timestamp_et").dt.date().unique().to_list()
    assert len(days) == 1
    assert str(days[0]) == start
