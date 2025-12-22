# Dataflow

## Metadata
- Generated: 2025-12-22T19:42:20Z
- Git SHA: a907a2eed87531d8178c3dc183d6f070182f9ebe
- Branch: codex/TICKET-000_project_state_rebuild
- Commands: `python tools/project_state_build.py`, `rg --files`, `sed -n '1,200p' README.md`, `sed -n '1,200p' docs/PROGRESS.md`, `sed -n '1,200p' CHANGELOG.md`, `sed -n '1,200p' pyproject.toml`, `sed -n '1,200p' Makefile`

## High-level flow (index ladders)
1) **Market data ingest**
   - Polygon index data via REST or WS: `src/kalshi_alpha/drivers/polygon_index/*` and `polygon_index_ws.py`.
   - Raw snapshots stored under `data/raw/...` (gitignored) or fixtures under `tests/data_fixtures/`.
2) **Calibration**
   - Index calibration outputs under `data/proc/calib/index/<symbol>/<horizon>/params.json`.
3) **Strategy PMFs**
   - `src/kalshi_alpha/strategies/index/*` produces PMFs aligned with ladders.
4) **Scanner**
   - `exec/runners/scan_ladders.py` aligns PMFs to strikes, applies fees and fill/slippage models, builds proposals.
5) **Risk gates**
   - PAL/VAR/drawdown via `core/risk/*` and `exec/limits.py`.
   - Freshness/quality gates via `configs/freshness.index.yaml` and `configs/quality_gates.index.yaml`.
6) **Execution (dry or live)**
   - Brokers in `brokers/kalshi/*` enforce maker-only + pilot boundaries; default is DRY.
7) **Reporting**
   - Markdown reports under `reports/<SERIES>/<YYYY-MM-DD>.md`.
   - Machine artifacts under `reports/_artifacts/`.

## Macro data flow (present but out of current trading scope)
- Drivers under `src/kalshi_alpha/drivers/*` for CPI, claims, treasury, nowcast, weather, AAA gas.
- Calibration outputs under `data/proc/*` used by macro strategies.
- Macro flows are often disabled in index-only runs via family scoping (`FAMILY=index`).

## Storage conventions
- `data/raw/`: raw snapshots (Polygon/Kalshi/feeds).
- `data/proc/`: processed tables, calibration params, ledger state.
- `reports/`: human-readable outputs; `_artifacts/` holds machine outputs.
- `tests/fixtures/` and `tests/data_fixtures/`: offline-safe fixtures to reproduce scans.

## Notable data artifacts
- `data/proc/ledger/*.jsonl`: paper/live ledger events.
- `data/proc/state/kill_switch`: presence forces NO-GO + cancel intent.
- `reports/_artifacts/go_no_go.json`: latest GO/NO-GO decision.
- `reports/_artifacts/monitors/*.json`: runtime monitors (freshness, sigma drift, EV gap).

## Data quality controls
- Freshness thresholds: `configs/freshness.yaml` (macro), `configs/freshness.index.yaml` (index).
- Fee/rule watcher: `monitor/fee_rules_watch.py` requires acknowledgements before scans.
- Basis audit: `tools/settlement_basis_audit.py` compares Polygon vs Kalshi settlement values.
