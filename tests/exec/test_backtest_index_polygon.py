import subprocess
from pathlib import Path

import polars as pl

from scripts.build_index_panel_polygon import build_panel
from kalshi_alpha.strategies.index.model_polygon import fit_from_panel, params_path, save_params


FIXTURE_ROOT = Path("tests/data_fixtures/index_panel_backtest/raw/polygon/index")


def test_backtest_cli_produces_trades(tmp_path):
    panel_path = tmp_path / "panel.parquet"
    panel = build_panel(
        symbols=("I:SPX", "I:NDX"),
        input_root=FIXTURE_ROOT,
        output_path=panel_path,
    )
    # Calibrate a minimal close model for INX.
    params_root = tmp_path / "calib"
    close_panel = panel.filter(pl.col("symbol") == "I:SPX")
    params = fit_from_panel(close_panel, horizon="close", symbols=["I:SPX"])
    save_params(params, params_path("INX", "close", root=params_root))

    trades_dir = tmp_path / "trades"
    reports_dir = tmp_path / "reports"
    cmd = [
        "python",
        "-m",
        "kalshi_alpha.exec.backtest_index_polygon",
        "--series",
        "INX",
        "--panel",
        str(panel_path),
        "--params-root",
        str(params_root),
        "--output-root",
        str(trades_dir),
        "--report-root",
        str(reports_dir),
        "--start-date",
        "2025-10-31",
        "--end-date",
        "2025-11-04",
        "--ev-threshold-cents",
        "0.5",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert result.returncode == 0
    csvs = list(trades_dir.glob("*.csv"))
    reports = list(reports_dir.glob("*.md"))
    assert csvs, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert reports
    frame = pl.read_csv(csvs[0])
    assert frame.height > 0
