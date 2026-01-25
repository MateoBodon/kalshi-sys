# Research Notes

## Metadata
- Generated: 2026-01-10T11:43:04Z
- Git SHA: 31316e59451269689f2da173d8a9c6d9049d3d5e
- Branch: codex/TICKET-111_project_state_refresh
- Commands: `python tools/project_state_build.py`, `python3 tools/agentic/project_state_refresh.py --zip`, `rg --files`, `sed -n '1,200p' README.md`, `sed -n '1,200p' docs/PROGRESS.md`, `sed -n '1,200p' CHANGELOG.md`, `sed -n '1,200p' pyproject.toml`, `sed -n '1,200p' Makefile`

## Scope framing
- Primary research focus: Kalshi index ladders (INX/INXU/NASDAQ100/NASDAQ100U) hourly + close windows.
- Macro strategies (CPI/claims/weather/teny) exist but are out of scope for current pilot decisions.
- See `docs/PLAN_OF_RECORD.md` and `kalshi_alpha_long_term_plan.md` for the official framing.

## Hypotheses being tested
- **Probability edge**: calibrated PMFs yield better odds than market-implied probabilities at window time.
- **Microstructure edge**: maker-first execution can capture EV after fees with realistic fill models.
- **Operational edge**: reliable, low-latency, ET-aligned quoting during windows yields stable execution.

## Evidence requirements (non-negotiable)
- Realistic fill curves derived from observed TOB snapshots and our own quote placements.
- Basis audit between Polygon index values and Kalshi settlement values per window.
- Freshness and calibration age gates pass for the series under test.
- Paper or live ledger evidence of fills with fees/slippage accounted.

## Primary research artifacts
- Basis audit: `tools/settlement_basis_audit.py` → `reports/settlement_basis/*.md`.
- Fill calibration data: `tools/build_fillcalib_dataset.py` → `reports/fillcalib/` + `data/proc/fillcalib/*`.
- Replay parity: `tools/replay.py` and `scripts/parity_gate.py` → `reports/_artifacts/replay*.parquet`.
- Scoreboards/readiness: `reports/scoreboard_7d.md`, `reports/pilot_readiness.md`.

## Working constraints
- Maker-only by default; taker routes require explicit flags and safety review.
- Fail-closed defaults: any ambiguity should block execution rather than allow it.
