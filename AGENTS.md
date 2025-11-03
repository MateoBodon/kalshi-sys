# AGENTS.md

## Purpose
Give coding agents the bare minimum to ship safe, correct changes fast: how to build, test, gate, and commit.

## Guardrails (global)
- Never commit secrets. Read env via `src/kalshi_alpha/utils/env.py`.
- CI must be green: `ruff`, `mypy`, `pytest`.
- Conventional Commits. Small PRs. No force-push.
- Touching risk/limits? Add/update tests and rationale in code & RUNBOOK.

## Build & Test
- Install: `pip install -e ".[dev]"`
- Lint/Type/Test: `ruff check . && mypy && pytest -q`
- Offline smoke:
  - `python -m kalshi_alpha.exec.pipelines.daily --mode teny_close --offline --driver-fixtures tests/fixtures --scanner-fixtures tests/data_fixtures --report`
- Scoreboard/readiness: `python -m kalshi_alpha.exec.scoreboard --window 7 --window 30`

## Live Safety (human-gated)
- Default broker is DRY. Live requires `--broker live` AND valid creds.
- Kill-switch file: `data/proc/state/kill_switch` halts submissions.
- Heartbeat must be <5 min old before placing any live order.

## TENY Ops (close window market)
- Window: 14:30–15:25 ET (Kalshi 10Y daily yield closes 4:25 PM ET).
- Required inputs: `treasury_yields` parquet fresh; macro dummies present; order-book imbalance available.
- Proposal policy: maker-only, 1-lots, ≤ 2 bins, EV_after_fees ≥ $0.05, truncated Kelly cap 0.25.
- Post-run: aggregate ledger, run readiness report, check ev_gap/fill_vs_alpha.

## Codex Workflow
1. Read this file and `README.md`, `docs/RUNBOOK.md`.
2. Plan minimal changes; implement with **small commits**.
3. Always run `ruff`, `mypy`, `pytest` locally before committing.
4. For reports/gates: add tests that break on divergence.

## Commit examples
- `feat(teny): add macro calendar dummies driver and wire into calibration`
- `fix(reports): plumb real go/no-go into markdown; add consistency test`
- `feat(fees): parse fee schedule pdf -> json; use in EV calc`
- `test(treasury): stricter DGS10 vs 10Y reconciliation cases`
- `docs(runbook): add TENY close ops checklist`
