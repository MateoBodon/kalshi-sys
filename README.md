# Kalshi Alpha Monorepo

Kalshi Alpha is a private research monorepo that consolidates reusable tooling for pricing, scanning, and backtesting Kalshi ladder markets. The codebase is structured as a Python namespace package under `kalshi_alpha` and is designed for dry-run order proposal generation with no authenticated order submission capabilities.

## Features

- Shared fee, pricing, and risk engines to support multiple Kalshi ladders (fees follow the October 1, 2025 round-up schedule using Decimal precision).
- Read-only public market-data client with offline fixture support for testing.
- Strategy stubs for CPI, jobless claims, Treasury yields, weather, and gasoline.
- Drivers for ingesting core macro datasets (BLS CPI, DOL ETA-539, Treasury par yields, NOAA/NWS Daily Climate Reports, AAA gas prices).
- Datastore snapshotters producing timestamped raw captures and processed Parquet tables via DuckDB.
- CLI runner (`kalshi-scan`) that scans ladders, applies strategies, fees, and PAL limits, and writes executable order proposal JSON (dry-run only).

## Repository Layout

- `src/kalshi_alpha/core/`: Shared libraries for fees, pricing, API access, risk, backtesting, and datastore management.
- `src/kalshi_alpha/drivers/`: Data ingestion helpers for macro datasets (fixtures provided for offline usage).
- `src/kalshi_alpha/strategies/`: Strategy modules returning distributions over Kalshi ladder bins.
- `src/kalshi_alpha/exec/`: Execution scaffolding, including runners, scanners, and broker placeholders.
- `configs/`: Configuration files (e.g., PAL policies).
- `data/`: Raw (`data/raw/`) and processed (`data/proc/`) data stores; both ignored by Git.
- `exec/proposals/`: Scanner outputs in JSON format (ignored by Git).
- `tests/`: Pytest suite with fixtures covering core subsystems.
- `notebooks/`: Research notebooks (not versioned for data).

## Tooling

- Python 3.11+
- Dependencies defined in `pyproject.toml` (PEP 621).
- Development tooling: pytest, hypothesis, ruff, mypy, pre-commit.
- `Makefile` targets simplify common workflows: formatting, linting, typing, testing, scanning.

## Quick Start

```bash
uv pip install -e ".[dev]"  # fall back to `pip` if `uv` is unavailable
make lint
make typecheck
make test
python -m kalshi_alpha.exec.runners.scan_ladders --series CPI --maker-only --min-ev 0.005 --dry-run
```

The CLI will read offline fixtures when network access is disabled and will write proposed dry-run orders into `exec/proposals/<series>/<YYYY-MM-DD>.json`. Additional options include `--strategy`, `--pal-policy`, `--max-loss-per-strike`, and adjacency controls via `--allow-tails`.

## References

- GPT-5-Codex prompting basics (minimal prompt, apply_patch editing workflow).
- Kalshi public market-data quick start (series, events, markets, orderbooks).
- Kalshi fee schedule (October 1, 2025) for fee formulas, round-up semantics, and S&P/Nasdaq half-rate path (no settlement fees).
- BLS CPI release schedule and methodology.
- DOL ETA-539 weekly initial claims reports.
- U.S. Treasury par yield curve (Daily Yield Curve Rates).
- NOAA/NWS Daily Climate Report as the definitive settlement source for weather ladders.
- AAA national and state gasoline price summaries (interface stubbed for future integration).

## License

Kalshi Alpha is released under the MIT License.
