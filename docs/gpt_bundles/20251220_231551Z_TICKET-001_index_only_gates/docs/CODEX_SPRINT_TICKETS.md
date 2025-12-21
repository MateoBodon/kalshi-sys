# CODEX SPRINT TICKETS — NEXT SPRINT (Index Ladders Only)

Sprint objective:
- Convert the current “NO-GO + no evidence” system into a credibility-ready PAPER system, then a safe PILOT candidate.
- Focus is realism + safety + instrumentation, not profits.

Hard scope:
- INX / INXU / NASDAQ100 / NASDAQ100U ladders only.
- No macro family work except to prevent macro from blocking index runs.

---

## Ticket #1 — Index-only GO/NO-GO gates (decouple from macro feed staleness)
Goal (1 sentence):
- Make index runs evaluate ONLY index-relevant quality/freshness/calibration gates, and stop blocking on unrelated macro feeds.

Likely files/modules:
- `configs/quality_gates.index.yaml`
- `configs/freshness.index.yaml`
- `src/kalshi_alpha/core/gates/quality_gates.py`
- `src/kalshi_alpha/exec/preflight_index.py`
- `src/kalshi_alpha/exec/runners/scan_ladders.py`
- Reporting integration:
  - `src/kalshi_alpha/exec/scoreboard.py`
  - `src/kalshi_alpha/exec/reports/ramp.py`

Acceptance criteria:
- Running an index supervisor/scanner with macro feeds stale still yields GO when:
  - Polygon WS freshness is within index thresholds AND
  - calibration freshness is within index thresholds AND
  - kill switch is off.
- GO/NO-GO reasons are series-scoped:
  - No macro namespaces appear in index-only runs.
- Scoreboard / readiness reports reflect the new logic:
  - no false NO-GO due to macro staleness.

Minimal tests/commands:
- `pytest -q`
- `python -m kalshi_alpha.exec.preflight_index` (should produce series-scoped GO/NO-GO)
- `python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run` (no false NO-GO)

Expected artifacts:
- `reports/_artifacts/go_no_go.json` updated (series-scoped)
- `docs/agent_runs/<RUN_NAME>/` run log bundle
- `docs/PROGRESS.md` entry
- `CHANGELOG.md` entry

---

## Ticket #2 — Settlement basis audit (Polygon vs Kalshi expiration value)
Goal (1 sentence):
- Quantify basis between Polygon and Kalshi expiration values at each traded window time; flag “strike flip risk.”

Likely files/modules:
- NEW: `tools/settlement_basis_audit.py`
- `src/kalshi_alpha/markets/discovery.py` (window → ticker mapping)
- `src/kalshi_alpha/core/kalshi_api.py` or public client wrapper used for settlement values
- `reports/settlement_basis/` (new)

Acceptance criteria:
- Command produces a daily report for any date + series:
  - basis distribution (mean/median/p95/p99)
  - “flip risk” flags where basis magnitude could change outcome near strikes.
- Reproducible:
  - saves raw inputs (Kalshi settlement value + Polygon snapshot values + timestamps)
  - can re-run without hidden live-only dependencies.

Minimal tests/commands:
- `python tools/settlement_basis_audit.py --day 2025-11-10 --series INXU`
- `python tools/settlement_basis_audit.py --day 2025-11-10 --series NASDAQ100U`
- Add at least one unit test with saved fixtures.

Expected artifacts:
- `reports/settlement_basis/<day>_<series>.md`
- `data/proc/settlement_basis/<day>_<series>.parquet` (or jsonl)
- run log + PROGRESS/CHANGELOG updates

---

## Ticket #3 — TOB snapshot logger + fill-calibration dataset skeleton
Goal (1 sentence):
- Start collecting TOB snapshots + our own quoting activity so fill probability can be measured (not guessed).

Likely files/modules:
- `src/kalshi_alpha/exec/telemetry/sink.py` (reuse if exists)
- `src/kalshi_alpha/exec/supervisor_index.py`
- `src/kalshi_alpha/brokers/kalshi/ws_client.py` (if needed for orderbook)
- NEW: `tools/build_fillcalib_dataset.py` (optional)
- `data/raw/kalshi/tob/` (new)
- `reports/fillcalib/` (new)

Acceptance criteria:
- For a dry-run supervisor session, snapshots are written:
  - timestamped
  - series/window labeled
  - depth-limited and size-bounded
- A dataset builder can convert snapshots to a derived table ready for calibration:
  - includes mid, bid/ask, depth, our quote price/size, time-to-expiry

Minimal tests/commands:
- `python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run --record-tob`
- `python tools/build_fillcalib_dataset.py --in data/raw/kalshi/tob --out data/proc/fillcalib/inxu.parquet`
- `pytest -q` (fixture-based tests for schema + bounds)

Expected artifacts:
- `data/raw/kalshi/tob/*.jsonl`
- `data/proc/fillcalib/*.parquet`
- `reports/fillcalib/README.md` (schema + how to use)
- run log + PROGRESS/CHANGELOG updates

---

## Ticket #4 — Pilot safety enforced at broker boundary (+ tests)
Goal (1 sentence):
- Ensure pilot constraints are enforced in the last-mile broker submit path (fail-closed), not only in strategy selection.

Likely files/modules:
- `configs/pilot.yaml`
- `src/kalshi_alpha/exec/runners/scan_ladders.py` (execute path)
- `src/kalshi_alpha/brokers/kalshi/live.py`
- `src/kalshi_alpha/core/execution/order_queue.py`
- `src/kalshi_alpha/exec/heartbeat.py`

Acceptance criteria:
- Live mode cannot run without explicit acknowledgement and correct environment.
- Pilot mode rejects:
  - crossing orders (maker-only)
  - orders outside window guard
  - size > caps, or too many concurrent ladders
- Kill switch blocks submits and replace/cancel spam.

Minimal tests/commands:
- `pytest -q`
- A small integration test that simulates a crossing order and expects rejection.
- `touch data/proc/state/kill_switch` + run pilot mode in dry environment → must not submit.

Expected artifacts:
- new/updated tests under `tests/`
- run log + PROGRESS/CHANGELOG updates

---

## Ticket #5 — AWS / 24-7 supervisor wiring (audit-ready runbook + watchdog)
Goal (1 sentence):
- Produce an AWS-ready operational runbook and minimal wiring templates for 24/7 `supervisor_index` with alerts.

Likely files/modules:
- `docs/runbooks/aws_supervisor_index.md` (new)
- `docs/runbooks/oncall_checks.md` (new)
- `configs/ops/*.yaml` (new or existing)
- optional templates:
  - `deploy/systemd/supervisor_index.service`
  - OR `deploy/ecs/taskdef_supervisor_index.json`

Acceptance criteria:
- Documented deployment plan that covers:
  - environment variables + secrets handling
  - restart policies
  - log routing
  - health checks (heartbeat + monitors)
  - alert conditions
- Includes “break-glass” procedures:
  - kill switch
  - cancel-all
  - disable live broker quickly

Minimal tests/commands:
- N/A (docs + config templates), but must include a local smoke command:
  - `python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run`

Expected artifacts:
- new runbooks
- run log + PROGRESS/CHANGELOG updates

---

## Out of scope (for this sprint)
- New macro strategies, CPI/claims/weather pipelines
- Scaling position sizes beyond pilot caps
- Any claims of profitability without real fill evidence + basis audit
