# CODEX SPRINT TICKETS — NEXT SPRINT (Index Ladders only)

Sprint intent:
- Unblock credible PAPER evidence for INX/NDX ladders.
- Build the measurement pipeline needed to calibrate execution realism.
- Remove “scope foot-guns” and tighten GO/NO-GO determinism.

Scope reminder:
- ONLY: INX / INXU / NASDAQ100 / NASDAQ100U ladders (hourly + daily close).
- Polygon-only feed.
- No macro systems.

---

## TICKET-101 — Decouple index-only GO/NO-GO from macro freshness + add explicit scope in artifacts

**Goal (1 sentence):** Index ladder runs must not fail due to macro freshness; go/no-go artifacts must declare `scope=index` and list only scoped blockers.

**Likely files/modules:**
- `src/kalshi_alpha/exec/monitors/freshness.py`
- `src/kalshi_alpha/core/gates/quality_gates.py`
- `configs/freshness.index.yaml`
- `configs/quality_gates.index.yaml`
- `src/kalshi_alpha/exec/runners/scan_ladders.py`
- `src/kalshi_alpha/exec/preflight_index.py`

**Acceptance criteria:**
- Running index-only entrypoints does **not** include macro feeds in freshness gating unless explicitly configured.
- go/no-go artifact includes:
  - `scope: "index"`
  - `scoped_blockers: [...]` (only index-scoped reasons)
- Existing pytest suite passes.
- Add at least one unit/integration test that simulates “macro stale, index fresh” and expects GO.

**Minimal tests/commands:**
- `pytest -q`
- `make monitors`
- `python -m kalshi_alpha.exec.preflight_index`
- `python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run`

**Expected artifacts:**
- `reports/paper/<YYYY-MM-DD>/go_no_go.json` contains `scope=index`
- `docs/agent_runs/<RUN_NAME>/` with `TESTS.md` and `DIFF.patch`

**Status:** DONE (2025-12-23, retry evidence captured)

---

## TICKET-102 — Settlement basis audit as first-class gate + daily artifact + strike flip risk summary

**Goal (1 sentence):** Basis audit becomes a required daily artifact and can fail-closed when missing/stale or risky.

**Likely files/modules:**
- `tools/settlement_basis_audit.py`
- `src/kalshi_alpha/exec/preflight_index.py`
- `src/kalshi_alpha/exec/monitors/runtime.py`
- `configs/quality_gates.index.yaml`
- `docs/PLAN_OF_RECORD.md`

**Acceptance criteria:**
- New artifact written per series/day:
  - basis distribution (quantiles)
  - per-window deltas
  - “flip risk” flags for likely strike spacing / quote distances
- Preflight or quality gate fails closed if basis audit missing/stale for the series/day.
- Add fixture-based test that validates output schema on a synthetic window.

**Minimal tests/commands:**
- `pytest -q`
- `python tools/settlement_basis_audit.py --help`
- `python -m kalshi_alpha.exec.preflight_index`

**Expected artifacts:**
- `reports/basis/<SERIES>/<YYYY-MM-DD>.md`
- `data/proc/basis/<SERIES>/<YYYY-MM-DD>.json`

**Status:** DONE (2025-12-26) — dry-run telemetry artifacts + ops volume report captured; retention pruning verified.

**Follow-up:** 2025-12-25 — REST snapshot switched to v3 indices endpoint + marketstatus guard added for closed/extended hours.
**Follow-up:** 2025-12-26 — Freshness monitor now honors marketstatus (closed/extended) + ops market status CLI added.

---

## TICKET-103 — Start bounded TOB + quote-intent telemetry capture for index ladders (PAPER-safe) + retention proof

**Goal (1 sentence):** During dry-run windows, collect TOB snapshots and quote intents with bounded size and clear correlation keys.

**Likely files/modules:**
- `src/kalshi_alpha/exec/collectors/tob_logger.py`
- `src/kalshi_alpha/exec/telemetry/sink.py`
- `src/kalshi_alpha/exec/telemetry/shipper.py`
- `src/kalshi_alpha/exec/supervisor_index.py`
- `src/kalshi_alpha/exec/runners/micro_index.py`
- `src/kalshi_alpha/exec/housekeep.py` (retention)
- `docs/DOCS_AND_LOGGING_SYSTEM.md`

**Acceptance criteria:**
- During dry-run window, TOB snapshots and quote intents are written with bounded depth/size.
- Telemetry rows include: `run_id`, `window_id`, `series`, `market_ticker`, `ts`.
- Housekeeping prevents unbounded growth and is documented (retention days + max bytes/window).

**Minimal tests/commands:**
- `pytest -q`
- `make collect-polygon-ws`
- `python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run --record-tob`

**Expected artifacts:**
- `data/proc/telemetry/tob/<RUN_ID>.jsonl.gz`
- `data/proc/telemetry/quote_intents/<RUN_ID>.jsonl.gz`
- `reports/ops/telemetry_volume_<DATE>.md`

**Status:** DONE (2025-12-23)

---

## TICKET-104 — Build fill calibration dataset + conservative maker fill curves + wire into defaults

**Goal (1 sentence):** Produce empirical maker fill curves from telemetry and use conservative defaults when sample sizes are small.

**Likely files/modules:**
- `tools/build_fillcalib_dataset.py`
- `src/kalshi_alpha/core/execution/fillprob.py`
- `src/kalshi_alpha/core/execution/fillratio.py`
- `src/kalshi_alpha/core/execution/index_models.py`
- `src/kalshi_alpha/replay/fill_model.py`
- `docs/PLAN_OF_RECORD.md`

**Acceptance criteria:**
- Tool produces dataset with sample counts and derived fill curve per series/window bucket.
- Scanner can load fill curves and reports `uncalibrated` only when no data exists.
- Conservative defaults used below minimum sample threshold.
- Adds at least one fixture-based unit test for curve generation logic.

**Minimal tests/commands:**
- `pytest -q`
- `python tools/build_fillcalib_dataset.py --help`
- `make pilot-readiness`

**Expected artifacts:**
- `data/proc/fillcalib/curves_<ASOF_DATE>.json`
- `reports/fillcalib/<ASOF_DATE>.md`

**Status:** DONE (2025-12-26)

---

## TICKET-105 — Calibration age visibility: single summary artifact + scoreboard + explicit NO-GO reasons

**Goal (1 sentence):** Make it impossible to ignore stale calibration by surfacing ages in a single committed artifact and in readiness/scoreboard outputs.

**Likely files/modules:**
- `src/kalshi_alpha/exec/pilot_readiness.py`
- `src/kalshi_alpha/exec/scoreboard.py`
- `jobs/calibrate_hourly.py`
- `jobs/calibrate_close.py`
- `docs/PLAN_OF_RECORD.md`

**Acceptance criteria:**
- One artifact lists calibration ages per series/horizon and flags stale items.
- Scoreboard renders calibration age status for each series.
- NO-GO reasons explicitly name which calibration file(s) are stale/missing.
- Adds at least one test for “stale calibration → NO-GO reason includes filename.”

**Minimal tests/commands:**
- `pytest -q`
- `make calibrate-index`
- `make pilot-readiness`

**Expected artifacts:**
- `reports/calibration/calibration_ages_<ASOF_DATE>.md`
- `reports/pilot_readiness_<ASOF_DATE>.md` includes calibration block

**Status:** DONE (2025-12-29)

---

## TICKET-106 — AWS supervisor wiring proof (PAPER): systemd + CloudWatch + crash recovery drill artifact

**Goal (1 sentence):** Demonstrate a 24/7 PAPER supervisor on AWS with logs/metrics landing in CloudWatch and a documented restart/reconciliation drill.

**Likely files/modules:**
- `configs/systemd/kalshi-*.service`
- `configs/systemd/kalshi-*.timer`
- `configs/cloudwatch/kalshi-supervisor-index.json`
- `configs/logrotate/kalshi-alpha`
- `docs/runbooks/aws_supervisor_index.md`
- `docs/runbooks/oncall_checks.md`
- `src/kalshi_alpha/exec/supervisor_index.py`
- `src/kalshi_alpha/sched/hotrestart.py`

**Acceptance criteria:**
- Supervisor runs continuously (dry-run) with auto-restart and writes heartbeats/monitor artifacts.
- CloudWatch receives logs; at least one heartbeat/freshness metric is emitted.
- Documented crash recovery drill: restart supervisor and prove safe reconciliation (no duplicate orders, no stale state).

**Minimal tests/commands:**
- `python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run`
- `make monitors`

**Expected artifacts:**
- `reports/ops/aws_supervisor_dryrun_<DATE>.md`
- `docs/agent_runs/<RUN_NAME>/` includes screenshots/log excerpts (redacted)

---

## TICKET-107 — Harden broker boundary: fail-closed enforcement for pilot caps + maker-only + kill switch (independent of strategy logic)

**Goal (1 sentence):** Ensure the broker layer enforces pilot caps, maker-only semantics, and kill switch behavior even if strategy proposes unsafe orders.

**Likely files/modules:**
- `src/kalshi_alpha/brokers/kalshi/live.py`
- `src/kalshi_alpha/brokers/kalshi/base.py`
- `src/kalshi_alpha/exec/limits.py`
- `src/kalshi_alpha/exec/runners/scan_ladders.py`
- `configs/pilot.yaml`
- `docs/PLAN_OF_RECORD.md`

**Acceptance criteria:**
- Live broker refuses to submit without explicit acknowledgement + required env vars.
- Crossing orders rejected in maker-only mode (post-only + price checks).
- Kill switch engaged causes no submits and emits a clear audit log event.
- Tests added/updated (live safety, limits, kill switch).

**Minimal tests/commands:**
- `pytest -q`
- `make live-smoke  # read-only`
- `python -m kalshi_alpha.exec.live_smoke`

**Expected artifacts:**
- `reports/safety/live_smoke_<DATE>.md`
- `docs/agent_runs/<RUN_NAME>/` with clear proof of enforced behavior

---

## TICKET-108 — GPT bundle completeness for calibration/readiness artifacts

**Goal (1 sentence):** Ensure per-ticket GPT bundles include fillcalib, pilot readiness, and calibration age artifacts, and fail-closed if ARTIFACTS.md lists files missing from the bundle.

**Likely files/modules:**
- `Makefile` (gpt-bundle target)
- `tools/verify_gpt_bundle.py`
- `tools/gpt_bundle_builder.py`
- `tests/test_gpt_bundle_verifier.py`
- `tests/test_gpt_bundle_builder.py`
- `docs/DOCS_AND_LOGGING_SYSTEM.md`

**Acceptance criteria:**
- Bundles include `data/proc/fillcalib/*.json`, `reports/fillcalib/*.md`, `reports/pilot_ready.json`, `reports/pilot_readiness.md`, and `reports/calibration/*.md` when present.
- Bundle verification fails if `docs/agent_runs/<RUN_NAME>/ARTIFACTS.md` lists an existing file missing from the bundle.
- Regression tests cover bundle contents and fail-closed behavior.

**Minimal tests/commands:**
- `pytest -q`
- `PYTHON=python3 make gpt-bundle TICKET=TICKET-108 RUN_NAME=<RUN_NAME>`

**Expected artifacts:**
- `docs/gpt_bundles/gpt_bundle_TICKET-108_<RUN_NAME>.zip`
- `docs/agent_runs/<RUN_NAME>/` updated with bundle evidence

**Status:** DONE (2025-12-26)
