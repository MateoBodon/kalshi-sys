## 1. Project snapshot (5–10 bullets)

* **FACT (what I used):** This audit is grounded in the attached snapshot bundle only:
  `docs/gpt_bundles/project_state_20251222_194751Z_a907a2e/PROGRESS.md`,
  `docs/.../docs/PLAN_OF_RECORD.md`, `docs/.../docs/DOCS_AND_LOGGING_SYSTEM.md`, and the `project_state/` suite (`ARCHITECTURE.md`, `PIPELINE_FLOW.md`, `DATAFLOW.md`, `CURRENT_RESULTS.md`, `EXPERIMENTS.md`, `KNOWN_ISSUES.md`, `OPEN_QUESTIONS.md`, `CONFIG_REFERENCE.md`, `SERVER_ENVIRONMENT.md`, `TEST_COVERAGE.md`, `MODULE_SUMMARIES.md`, `FUNCTION_INDEX.md`, `DEPENDENCY_GRAPH.md`, plus `_generated/{repo_inventory.json,symbol_index.json,...}`).
* **FACT:** Snapshot metadata pins the repo to **Git SHA `a907a2eed87531d8178c3dc183d6f070182f9ebe`** (branch `codex/TICKET-000_project_state_rebuild`). (`project_state/INDEX.md`)
* **FACT:** Hard scope is **index ladders only**: **INX / INXU / NASDAQ100 / NASDAQ100U**, hourly + daily close, Polygon-only feed. (`docs/PLAN_OF_RECORD.md` header + §1)
* **FACT:** Current gate status is explicitly **PAPER** (not PILOT/LIVE). (`PROGRESS.md` → “Gate status”)
* **FACT:** There is a full scan→propose→(dry/live adapter) surface for index ladders via:
  `kalshi_alpha.exec.runners.scan_ladders`, `kalshi_alpha.exec.preflight_index`, `kalshi_alpha.exec.supervisor_index`, plus hourly/close scanners. (`project_state/PIPELINE_FLOW.md` → “Primary entrypoints (index scope)” + “Index supervisor flow”)
* **FACT:** The most important “where we really are” evidence is **negative**:
  **no committed evidence of realized fills**, and pilot readiness is **NO‑GO for 0/4 series due to `insufficient_data` with `fills=0`**. (`project_state/CURRENT_RESULTS.md` → “Pilot readiness” + “Summary”)
* **FACT:** Safety posture is explicitly “fail‑closed” with a kill switch file and explicit acknowledgements for live. (`project_state/DATAFLOW.md` → “Notable data artifacts”; `project_state/STYLE_GUIDE.md` → “Repo-specific safety conventions”; `docs/PLAN_OF_RECORD.md` → §2.5)
* **FACT:** Test coverage is non-trivial by breadth (≈167 pytest files, multiple safety/risk/fees areas). (`project_state/TEST_COVERAGE.md`)
* **UNCLEAR (critical limitation):** This bundle does **not** include actual source/config contents (only inventories + symbol/function indices). So I can reconstruct flows and interfaces, but I cannot verify implementation details inside the code or the current values inside `configs/*.yaml|json` beyond what’s described in docs/indices.

---

## 2. System framing (what we’re trading + edge hypothesis)

### Instruments and settlement mapping

* **FACT:** Traded instruments are **Kalshi index ladder contracts** for:

  * **Hourly intraday (“U” series):** INXU, NASDAQ100U
  * **Daily close:** INX, NASDAQ100
    (`docs/PLAN_OF_RECORD.md` → §1.1; `project_state/PIPELINE_FLOW.md` → “Index supervisor flow (hourly + close)”)
* **FACT (repo mechanism):** Market/window identification is intended to be programmatic via:

  * **Scheduler windows (ET aware):** `src/kalshi_alpha/sched/windows.py` (“US/Eastern-aware scheduler for hourly and close index ladder windows.”)
  * **Market discovery:** `src/kalshi_alpha/markets/discovery.py` (“Market discovery utilities for INX/NDX ladders.”)
  * **Rule semantics:** `src/kalshi_alpha/config/index_rules.py` (“Load index ladder rule semantics from the markdown rulebook.”)
    (`project_state/MODULE_SUMMARIES.md`; `project_state/FUNCTION_INDEX.md`)
* **FACT (non-negotiable):** The docs correctly treat **Polygon as an input feed, not settlement truth**; settlement comes from Kalshi-defined expiration value. Therefore basis audit is a required gate. (`docs/PLAN_OF_RECORD.md` → §1.2; `project_state/DATAFLOW.md` → “Basis audit”; `tools/settlement_basis_audit.py` module doc in `project_state/_generated/symbol_index.json`)
* **ASSUMPTION (because code/config not included):** The specific ladder types you intend to trade are the “Above/Below” and “Range” ladders mentioned in the user prompt and referenced by strategy modules (`hourly_above_below.py`, `close_range.py`). I cannot confirm every ticker/contract schema mapping without the rulebook markdown + discovery outputs.

### Decision cadence (hourly + close)

* **FACT:** Cadence is window-based:

  * Supervisor picks active/next window (`src/kalshi_alpha/exec/supervisor_index.py` functions `_pick_window`, `_series_to_run`, `_run_loop`).
  * Micro-runner executes a single window with “1‑lot maker quotes” (`src/kalshi_alpha/exec/runners/micro_index.py` module doc).
    (`project_state/FUNCTION_INDEX.md`; `project_state/MODULE_SUMMARIES.md`)

### Edge hypothesis (what is supposed to create EV after fees)

* **FACT (documented hypotheses):**

  * **Probability edge:** your PMF/CDF is better calibrated than market-implied odds.
  * **Microstructure edge:** maker-first capture without paying taker costs (requires real fill probability + queue realism).
  * **Operational edge:** you quote reliably + correctly in the right windows.
    (`project_state/RESEARCH_NOTES.md` → “Hypotheses being tested”; `docs/PLAN_OF_RECORD.md` → §1.3)
* **ASSUMPTION (trader framing):** The “edge” is expected to materialize as **positive expected value on specific ladder bins** where:

  * your model-implied probability differs from market-implied probability,
  * **and** after applying **fees + slippage + a conservative fill model**, the **EV remains > 0** at your intended quote price/side.
    This is consistent with the presence of `core/pricing.expected_value_after_fees`, `exec/quote_optim`, and fill/slippage modules, but the bundle doesn’t include the actual pricing policy parameters.

### Required realities for profitability (all must hold)

* **FACT (must be true per docs):**

  * **Fee correctness:** schedule + rounding correct; maker vs taker classification not hand-waved. (`docs/PLAN_OF_RECORD.md` → §1.4; `core/fees/*`, `exec/fees.py`)
  * **Fill realism:** maker fill probabilities must be empirically measured (TOB + your quotes), not guessed. (`docs/PLAN_OF_RECORD.md` → §1.4; `exec/collectors/tob_logger.py`, `tools/build_fillcalib_dataset.py`, `core/execution/fillprob.py`)
  * **Latency + timing:** ET-alignment and bounded clock skew. (`docs/PLAN_OF_RECORD.md` → §1.4; `scan_ladders._clock_skew_seconds`; `brokers/kalshi/http_client.KalshiClockSkewError`)
  * **Calibration stability:** sigma_tod / drift and their ages must be monitored and gated. (`docs/PLAN_OF_RECORD.md` → §1.4; `jobs/calibrate_*`; `exec/monitors/sigma_drift.py`; `exec/pilot_readiness.calibration_age_days`)
  * **Risk caps:** PAL/VAR/drawdown enforced; pilot is maker-only + tiny size. (`docs/PLAN_OF_RECORD.md` → §1.4 + §3; `core/risk/*`, `risk/var_index.py`, `risk/correlation.py`, `core/risk/drawdown.py`)
  * **Ops:** 24/7 supervisor, crash recovery, alerting, and kill switch must be proven with artifacts. (`docs/PLAN_OF_RECORD.md` → §1.4 + §2.6; `exec/supervisor.py`, systemd + CloudWatch configs referenced in `project_state/CONFIG_REFERENCE.md`)

---

## 3. Pipeline reconstruction (end-to-end)

Below is the **actual pipeline as represented by interfaces + docs** in this snapshot. Where I can’t verify implementation details (because source is not included), I mark it.

### A) Scheduling and window selection

* **Key modules/files**

  * `src/kalshi_alpha/sched/windows.py` (ET-aware index windows)
  * `src/kalshi_alpha/exec/window_guard.py` (guard runners to windows)
  * `src/kalshi_alpha/exec/supervisor_index.py` (24/7 loop around windows)
    (`project_state/MODULE_SUMMARIES.md`; `project_state/FUNCTION_INDEX.md`)
* **Key invariants**

  * **ET correctness** (DST-safe) and “window” boundaries match Kalshi listing/settlement rules.
  * Supervisor must be able to pick “active or next upcoming” window (`_pick_window`). (`FUNCTION_INDEX.md`)
* **Likely failure modes**

  * DST/weekend misalignment → quoting wrong contracts or missing windows.
  * Multiple supervisors overlap → duplicated/crossed quotes.
* **Tested vs untested**

  * **FACT:** Index runners/scanners have tests listed (e.g., `tests/test_micro_runner.py`, `tests/test_index_scanners.py`). (`project_state/TEST_COVERAGE.md`)
  * **UNCLEAR:** exact DST boundary tests and lockout semantics without reading tests.

### B) Preflight GO/NO‑GO gating

* **Key modules/files**

  * `src/kalshi_alpha/exec/preflight_index.py` (GO/NO‑GO checks)
  * `src/kalshi_alpha/exec/heartbeat.py` (heartbeat + kill switch)
  * `src/kalshi_alpha/core/gates/quality_gates.py` (quality gates)
    (`project_state/PIPELINE_FLOW.md` step 2; `FUNCTION_INDEX.md` → `run_preflight`)
* **Key invariants**

  * Kill switch engaged ⇒ **NO‑GO**. (`heartbeat.kill_switch_engaged`; `DATAFLOW.md` notes `data/proc/state/kill_switch`)
  * Calibration not stale (`_calibration_check` with `max_age_days`). (`FUNCTION_INDEX.md`)
  * Polygon connectivity health (`_polygon_ping`). (`FUNCTION_INDEX.md`)
  * Live broker credentials required when `require_kalshi=True` (default). (`preflight_index.run_preflight` signature)
* **Likely failure modes**

  * Gate scope coupling: index runs blocked by unrelated macro feed staleness. (**Called out as a blocker in `docs/PLAN_OF_RECORD.md` → “Top blockers”.**)
  * “Require_kalshi” semantics accidentally too strict for paper-only environments (or vice versa).
* **Tested vs untested**

  * **FACT:** Freshness gate tests exist. (`project_state/TEST_COVERAGE.md` → “Monitors + freshness”)
  * **UNCLEAR:** Whether the index-only scope is defaulted and enforced in production entrypoints (this is explicitly an open question). (`project_state/OPEN_QUESTIONS.md` → “Configuration clarity”)

### C) Data ingest and freshness enforcement (Polygon + WS sentry)

* **Key modules/files**

  * Polygon WS singleton + metrics: `src/kalshi_alpha/drivers/polygon_index_ws.py`
  * Websocket freshness sentry: `src/kalshi_alpha/data/ws_sentry.py`
  * WS collector: `src/kalshi_alpha/exec/collectors/polygon_ws.py`
  * Freshness monitor + artifact: `src/kalshi_alpha/exec/monitors/freshness.py`
    (`project_state/PIPELINE_FLOW.md` step 3; `DATAFLOW.md`)
* **Key invariants**

  * Final-minute freshness: sentry must block if WS is stale near decision time. (`ws_sentry` module doc; `PIPELINE_FLOW.md`)
  * Timestamps must be UTC→ET safe. (`freshness._ensure_utc`, `ws_sentry._ensure_utc`)
* **Likely failure modes**

  * WS disconnects/reconnect storms near windows → stale data and bad/no-go decisions.
  * Polygon latency spikes not reflected in gating thresholds (false GO).
* **Tested vs untested**

  * **FACT:** Freshness gating has tests (per `TEST_COVERAGE.md`).
  * **UNCLEAR:** Whether WS freshness is enforced consistently in **all** index entrypoints vs only supervisor.

### D) Market discovery (Kalshi public data) + strike grid extraction

* **Key modules/files**

  * Read-only market client: `src/kalshi_alpha/core/kalshi_api/__init__.py` (Series/Event/Market/Orderbook)
  * Index market discovery: `src/kalshi_alpha/markets/discovery.py`
    (`project_state/MODULE_SUMMARIES.md`; `PIPELINE_FLOW.md` mentions “discovery”)
* **Key invariants**

  * Correct mapping: trading-day window ↔ Kalshi event ↔ market tickers ↔ strike ladder.
  * No silent “orphan markets” or wrong event matching. (`markets/discovery.py` has `_orphan_markets`, `_match_for_window`)
* **Likely failure modes**

  * Kalshi ticker format changes → `_infer_close_from_ticker` breaks.
  * You discover wrong window’s ladder → systematic losses / false edge.
* **Tested vs untested**

  * **UNCLEAR:** Specific discovery parity tests exist? Not shown in included docs. (Plan-of-record calls for discovery parity artifacts, but those aren’t included here.)

### E) Model/strategy PMF generation (Polygon-only index distribution)

* **Key modules/files**

  * Strategy PMFs:

    * `src/kalshi_alpha/strategies/index/hourly_above_below.py`
    * `src/kalshi_alpha/strategies/index/close_range.py`
    * shared CDF/calibration: `src/kalshi_alpha/strategies/index/cdf.py`
  * Model params access: `src/kalshi_alpha/models/pmf_index.py` and `src/kalshi_alpha/strategies/index/model_polygon.py`
  * Calibration jobs: `jobs/calibrate_hourly.py`, `jobs/calibrate_close.py`, etc. (`project_state/PIPELINE_FLOW.md` → “Calibration jobs”)
* **Key invariants**

  * Calibration file freshness is enforced (preflight + readiness).
  * PMF aligns to the actual strikes/rungs traded (alignment step follows).
* **Likely failure modes**

  * Calibration drift intraday (sigma_tod wrong) → systematic mispricing.
  * PMF tail behavior wrong → catastrophic around tails/halts.
* **Tested vs untested**

  * **FACT:** Backtest/replay tooling exists and is tested broadly. (`TEST_COVERAGE.md` → “Replay/backtest”)
  * **UNCLEAR:** Whether calibration diagnostics (PIT, CRPS) are being produced and gated in committed artifacts (explicitly cited as missing). (`KNOWN_ISSUES.md` → “Calibration freshness… not surfaced in a single committed summary”)

### F) Pricing/EV computation (fees + slippage + fills)

* **Key modules/files**

  * Pricing utilities: `src/kalshi_alpha/core/pricing/*` (`expected_value_after_fees`, `yes_no_expected_values`, etc.)
  * Fee schedule: `src/kalshi_alpha/core/fees/__init__.py`, `src/kalshi_alpha/exec/fees.py`
  * Fill/slippage models: `src/kalshi_alpha/core/execution/fillprob.py`, `fillratio.py`, `slippage.py`, `index_models.py`
    (`MODULE_SUMMARIES.md`; `FUNCTION_INDEX.md`; `PIPELINE_FLOW.md` step 5)
* **Key invariants**

  * Fees are applied with correct rounding (there are explicit rounding helpers in fee modules).
  * Fill model is conservative unless calibrated; slippage model is conservative unless calibrated.
* **Likely failure modes**

  * Misclassifying maker vs taker fees (or ignoring taker paths entirely).
  * Fill model too optimistic → “edge” evaporates after reality.
* **Tested vs untested**

  * **FACT:** Fee/slippage have tests per `TEST_COVERAGE.md`.
  * **UNCLEAR:** How “maker-only” is enforced at the last mile (broker API flags + crossing checks) without code.

### G) Opportunity selection, proposal construction, sizing, and honesty gates

* **Key modules/files**

  * Main scanner: `src/kalshi_alpha/exec/runners/scan_ladders.py` (proposal selection, honesty gates, report writing, optional broker execution). (`FUNCTION_INDEX.md` shows the whole surface.)
  * Alignment/mispricing: `core/pricing/align.py`, `core/pricing/mispricing.py`
  * Quote optimization: `exec/quote_microprice.py`, `exec/quote_optim.py`
  * Sizing: `core/sizing/kelly.py`, `config/size_ladder.py`
* **Key invariants**

  * Only propose orders that pass:

    * min EV threshold (`scan_series(min_ev=...)`)
    * probability sanity (sum gaps / monotonic survival checks like `_is_monotone`, `prob_sum_gap_threshold`)
    * EV honesty constraints (`_load_ev_honesty_constraints`, `_apply_ev_honesty_gate`)
* **Likely failure modes**

  * “Honesty gate” depends on replay/constraints that aren’t present → false confidence or overly strict NO‑GO.
  * Quote optimization penalties (microprice, freshness widening) can silently change your effective edge.
* **Tested vs untested**

  * **FACT:** There are monitors and sequential guard modules (`exec/monitors/sequential.py`) to catch EV deltas.
  * **UNCLEAR:** Whether proposal selection is robust to sparse liquidity (no TOB) without seeing implementation.

### H) Risk controls and broker execution

* **Key modules/files**

  * Risk primitives: `core/risk/*` (PALGuard, PortfolioRiskManager), `risk/var_index.py`, `risk/correlation.py`, `core/risk/drawdown.py`
  * Limit enforcement: `exec/limits.py` (ProposalLimitChecker)
  * Brokers: `brokers/kalshi/dry.py` and `brokers/kalshi/live.py` (has `_validate_live_environment`)
  * Order state: `exec/state/orders.py`, `core/execution/order_queue.py`, `sched/hotrestart.py`
* **Key invariants**

  * Broker submission path must be **fail-closed**:

    * kill switch engaged ⇒ no submits,
    * pilot caps enforced,
    * maker-only enforced (no crossing).
  * Risk caps should be enforced *before* and *at* broker layer (defense in depth).
* **Likely failure modes**

  * Strategy code passes bad order; broker wrapper fails open.
  * Restart loses track of live orders; duplicates or leaves stale exposure.
* **Tested vs untested**

  * **FACT:** tests exist around broker live safety and limits (`tests/test_broker_live_safety.py`, `tests/test_limits.py` listed in `TEST_COVERAGE.md`).
  * **UNCLEAR:** Exact kill-switch cancel behavior for live orders (I see “cancel intent” language in docs, but can’t confirm implementation).

### I) Monitoring, reporting, and post-trade evaluation / replay

* **Key modules/files**

  * Runtime monitors: `exec/monitors/runtime.py` (ev_gap, auth errors, kill switch, ws disconnects, drawdown, freeze windows)
  * Scoreboards: `exec/scoreboard.py`
  * Pilot readiness: `exec/pilot_readiness.py`, ramp readiness: `exec/reports/ramp.py`
  * Telemetry: `exec/telemetry/sink.py`, `exec/telemetry/shipper.py`
  * Replay/parity: `core/archive/replay.py`, `tools/replay.py`, `scripts/parity_gate.py`
* **Key invariants**

  * Every run emits artifacts that can be audited (go/no-go artifact, monitor artifacts, proposals, ledger).
  * Post-trade evaluation must not be based on “I ran it locally”; must be reproducible artifacts. (`docs/DOCS_AND_LOGGING_SYSTEM.md` → “No invisible state”)
* **Likely failure modes**

  * Artifacts not persisted on AWS disk → you can’t diagnose.
  * Telemetry too large/noisy → disk fill, degraded performance.
* **Tested vs untested**

  * **FACT:** Replay/backtest tests exist (`tests/test_parity_gate.py`, etc. found in `repo_inventory.json`).
  * **UNCLEAR:** Whether alerting is wired (there’s `_maybe_notify` in monitors CLI, but no proof artifacts included).

---

## 4. Experimental evidence (what’s been run + how trustworthy)

### What has ACTUALLY been run (per included artifacts)

* **FACT:** There is **no `experiments/` directory**; research outputs are “ad-hoc” under `reports/`, `report/`, `data/proc/` per docs. (`project_state/EXPERIMENTS.md`)
* **FACT:** Snapshot claims “latest” scoreboards exist but show **no ledger data** for both 7d and 30d windows. (`project_state/CURRENT_RESULTS.md` → “Scoreboards”)
* **FACT:** Pilot readiness report (14-day) is **GO series 0/4**, all NO‑GO due to `insufficient_data` with `fills=0`. (`project_state/CURRENT_RESULTS.md`)
* **UNCLEAR (important):** The referenced artifacts (`reports/scoreboard_*.md`, `reports/pilot_readiness.md`, `reports/<SERIES>/<DATE>.md`, backtest metric files) are **not included** in this bundle, and `project_state/_generated/repo_inventory.json` contains **0 `reports/` files** (likely because they are gitignored / local-only). So I cannot validate the underlying metrics, only the snapshot’s statements about them.

### EV/alpha claims: are they modeled with realistic fills/slippage/fees?

* **FACT:** The system has explicit machinery for:

  * fees (`core/fees/*`, `exec/fees.py`),
  * fill probability (`core/execution/fillprob.py`),
  * fill ratio tuning (`core/execution/fillratio.py`),
  * slippage (`core/execution/slippage.py`),
  * and “EV honesty constraints” (`scan_ladders._load_ev_honesty_constraints`, `_apply_ev_honesty_gate`).
    (`project_state/MODULE_SUMMARIES.md`; `project_state/FUNCTION_INDEX.md`)
* **FACT:** There is also a **“lightweight maker fill probability heuristic”** for index ladders (`strategies/index/fill_model.py` module doc), and a **“Polygon-only backtest harness”** (`strategies/index/backtest_polygon.py` module doc). This is a red flag unless clearly labeled as *toy/diagnostic*, because it can easily drift into “pretend fills.” (`project_state/_generated/symbol_index.json`)
* **UNCLEAR:** No included artifact demonstrates that fills/slippage/fees in any backtest were calibrated against **measured maker fills** (because fills are currently 0 per readiness). Therefore, **any EV estimate is “not yet credible”** for live decision-making until fill/basis are measured.

### Top 5 ways a backtest here could be accidentally optimistic (repo-specific)

1. **Fill optimism via heuristics:**
   `strategies/index/fill_model.py` is explicitly a heuristic; `core/execution/fillratio.py` uses visible-depth heuristics; if those are used without conservative calibration, your EV is fantasy.
2. **Maker-vs-taker mismatch:**
   Backtest/scoring utilities mention `_index_taker_fee` (`backtest/scoring.py`), while execution logic is “maker-first.” If you score with one fee regime and trade in another, you can “win” on paper and lose live.
3. **Settlement basis ignored:**
   Docs themselves say Polygon ≠ settlement truth and require basis audits (`docs/PLAN_OF_RECORD.md` → §1.2). Any backtest that settles on Polygon prints without measuring “Kalshi expiration value” mismatch is structurally biased.
4. **Time alignment + clock skew leakage:**
   `scan_ladders` has `_clock_skew_seconds`, and the Kalshi HTTP client has `KalshiClockSkewError`. If historical evaluation assumes perfect timestamps but production has skew, you will systematically quote wrong.
5. **Liquidity/queue position ignored:**
   “Maker-only” edge depends on queue priority and cancel/replace behavior. Without modeling replacement throttling (`exec/quote_microprice.ReplacementThrottle`) and queue, your fill estimates are likely too high and adverse selection too low.

### 3–6 targeted “debug experiments” to validate realism fast

All of these are **PAPER-safe** and aimed at killing false edges quickly.

* **Experiment 1: Basis audit on real windows (must become a gate)**

  * Run: `python tools/settlement_basis_audit.py --series INXU NASDAQ100U INX NASDAQ100 --day <recent>` (exact CLI unknown; module exists).
  * Output required: per-window **basis distribution** + “strike flip risk” near typical quoting distances.
  * Why: if basis noise is comparable to your edge, stop.
* **Experiment 2: Maker fill curve from TOB + your quote intents**

  * Collect bounded TOB snapshots via `exec/collectors/tob_logger.py` + telemetry sink.
  * Build dataset via `tools/build_fillcalib_dataset.py`.
  * Output required: empirical fill probability vs price level/time-to-expiry with sample sizes.
* **Experiment 3: Replay parity check for EV honesty**

  * Use `tools/replay.py` + `scripts/parity_gate.py` to replay archived manifests and verify proposal EV recomputes exactly.
  * Output required: “proposal EV parity” report; any drift blocks pilot.
* **Experiment 4: Fee-rule drift test**

  * Run `make fee-rules-watch` (exists in `make_targets.txt`) and ensure it blocks until acknowledged (per `CONFIG_REFERENCE.md` “Fee/rule watcher”).
  * Output required: artifact showing hash match and the config version used in scan.
* **Experiment 5: End-to-end offline reproducibility**

  * Run offline scanner entrypoints with fixtures (`PIPELINE_FLOW.md` examples) and ensure deterministic outputs across runs.
  * Output required: stable proposal set + stable monitor artifacts.
* **Experiment 6: Clock skew guardrail**

  * Force skew in a test harness (or simulate via injected clock if supported) and confirm `scan_ladders`/client fail closed and emits a clear NO‑GO reason.

---

## 5. Execution & risk controls (paper/pilot/live readiness)

### What gates prevent accidental live trading (as represented here)

* **FACT:** Default mode is “dry/paper” and live requires explicit conditions. (`docs/PLAN_OF_RECORD.md` → §2.5; `project_state/STYLE_GUIDE.md`)
* **FACT:** There is a **kill switch file** (`data/proc/state/kill_switch`) that forces NO‑GO and “cancel intent.” (`project_state/DATAFLOW.md` → “Notable data artifacts”; `exec/heartbeat.py`)
* **FACT:** Live broker includes an explicit environment validator (`brokers/kalshi/live.py` function `_validate_live_environment`). (`project_state/FUNCTION_INDEX.md`)
* **FACT:** Preflight can require Kalshi credentials (`preflight_index.run_preflight(... require_kalshi=True ...)`). (`project_state/FUNCTION_INDEX.md`)
* **FACT:** There are explicit “broker guards” inside the scanner entrypoint: `_enforce_broker_guards`, `_quality_gate_for_broker`, and a broker executor `execute_broker(...)`. (`project_state/FUNCTION_INDEX.md` → `scan_ladders.py` section)
* **FACT:** Window guard helpers exist (`exec/window_guard.guard_series_window`). (`project_state/FUNCTION_INDEX.md`)
* **FACT:** Pilot scaffolding exists (`exec/pilot/config.py`, `exec/pilot/runtime.py`, `configs/pilot.yaml` referenced). (`project_state/CONFIG_REFERENCE.md`; `symbol_index.json`)

### Are controls sufficient for a tiny pilot (1-lot maker-only)?

* **FACT (positive):** Risk stack is layered:

  * PAL policy (`core/risk`), proposal limit checker (`exec/limits.py`),
  * VaR and correlation limiters (`risk/var_index.py`, `risk/correlation.py`),
  * drawdown guard (`core/risk/drawdown.py`),
  * replacement throttling (`exec/quote_microprice.py`).
    This is directionally correct for a high-risk trading system.
* **FACT (blocking):** Pilot readiness is NO‑GO with fills=0; you don’t have the empirical dataset the system claims to require before pilot. (`CURRENT_RESULTS.md`; `PLAN_OF_RECORD.md` §3 “Gate B — PILOT”)
* **UNCLEAR (safety-critical):** Whether the **last-mile broker submit** is guaranteed to enforce:

  * maker-only (post-only) semantics,
  * max lot / max bins constraints,
  * kill-switch cancellation of live orders,
    without relying on strategy code behaving.
    Tests are referenced (`tests/test_broker_live_safety.py`), but their contents aren’t in the bundle.

### What’s missing (minimum for PILOT safety)

* **Missing proof artifacts (not implementation):**

  * A demonstrated **paper stability streak** (≥5 market days) with no violations. (`docs/PLAN_OF_RECORD.md` → “Gate B — PILOT Entry criteria”)
  * A real **fill dataset** and derived conservative fill curves. (`OPEN_QUESTIONS.md` → “Execution evidence gaps”)
  * A per-window **basis audit** for the exact windows you’ll pilot. (`OPEN_QUESTIONS.md`)
  * A “live order reconciliation after restart” proof (hotrestart exists, but no artifacts shown). (`sched/hotrestart.py` module doc)

---

## 6. AWS / ops readiness (audit)

### What exists (good signs)

* **FACT:** Repo includes systemd templates + CloudWatch agent config + logrotate:

  * `configs/systemd/*.service|*.timer` (10 files)
  * `configs/cloudwatch/kalshi-supervisor-index.json`
  * `configs/logrotate/kalshi-alpha`
    (`project_state/CONFIG_REFERENCE.md`; `project_state/_generated/repo_inventory.json`)
* **FACT:** Runbooks exist in repo (but not included in this bundle):
  `docs/runbooks/aws_supervisor_index.md`, `hourly.md`, `eod.md`, `oncall_checks.md`, `outage_playbook.md`, `postmortem_template.md`. (`repo_inventory.json`)
* **FACT:** There is an explicit 24/7 supervisor daemon (`exec/supervisor.py`) and an index-specific supervisor (`exec/supervisor_index.py`). (`MODULE_SUMMARIES.md`)
* **FACT:** There is explicit telemetry infrastructure (`exec/telemetry/sink.py`) with sanitization/depth-limiting helpers. (`FUNCTION_INDEX.md`)

### Ops foot-guns / gaps (what would break 24/7)

* **FACT:** “Which production environment is authoritative (local vs AWS)” is explicitly open. (`OPEN_QUESTIONS.md` → “Ops / deployment”)
* **FACT:** Clock skew has surfaced as an operational signal (“clock-skew exceeded” is mentioned as a known issue). (`KNOWN_ISSUES.md` → “Operational signals”; also `KalshiClockSkewError` exists)
* **UNCLEAR:** Alerting is not demonstrated. There’s a `_maybe_notify` hook in `exec/monitors/cli.py`, but no proof of Slack/email wiring or CloudWatch alarms in artifacts. (`FUNCTION_INDEX.md`)
* **UNCLEAR:** Crash recovery and idempotency are suggested (hotrestart, outstanding order state), but not proven by artifacts in this bundle.
* **ASSUMPTION (common AWS failure mode):** If telemetry/log volume isn’t bounded, disk will fill and the supervisor will degrade silently; you need hard retention + alarms. Housekeeping exists (`exec/housekeep.py`), but proof-of-operation is missing.

---

## 7. Biggest gaps, failure modes, and realism risks

Ranked by “can kill the project” severity:

1. **No fill evidence → no calibrated fill model → EV is not credible.**

   * **FACT:** fills=0, pilot readiness NO‑GO. (`CURRENT_RESULTS.md`)
2. **Settlement basis uncertainty can flip outcomes near strikes.**

   * **FACT:** docs demand basis audit; it is not shown as current evidence. (`PLAN_OF_RECORD.md` §1.2; `OPEN_QUESTIONS.md`)
3. **Gate scope coupling can block index trading for irrelevant reasons.**

   * **FACT:** explicitly listed as a top blocker. (`docs/PLAN_OF_RECORD.md` → “Top blockers”)
4. **Clock skew / timestamp alignment risk.**

   * **FACT:** system explicitly models skew errors and reports skew issues. (`brokers/kalshi/http_client.py`; `KNOWN_ISSUES.md`)
5. **Maker-only execution realism risk (queue/cancel/replace).**

   * **FACT:** modules exist for throttling and order state, but there is no empirical validation included. (`exec/quote_microprice.py`; `core/execution/order_queue.py`; `OPEN_QUESTIONS.md`)
6. **Config clarity risk (fees source of truth; FAMILY scoping).**

   * **FACT:** called out as open questions. (`OPEN_QUESTIONS.md`)
7. **“Too much surface area” risk (macro code exists).**

   * **FACT:** scope risk is explicitly noted: macro code could run if scoping misconfigured. (`KNOWN_ISSUES.md` → “Scope risk”)

---

## 8. Continue / pivot assessment (with decision criteria)

### Blunt assessment

* **FACT:** You have a serious **systems scaffold** (preflight, gating, supervisor, risk modules, telemetry, tests). (`project_state/PIPELINE_FLOW.md`; `MODULE_SUMMARIES.md`; `TEST_COVERAGE.md`)
* **FACT:** You do **not** yet have the *one thing that matters*: **measured execution reality** (fills + basis) for these ladders. (`CURRENT_RESULTS.md`; `OPEN_QUESTIONS.md`)
* **Conclusion:** **Continue**, but **only** if you treat the next phase as **measurement + realism validation**, not “alpha hunting.”
  Any “profitability talk” before fill/basis/fee reconciliation is premature.

### Decision criteria (continue vs pivot)

Continue if, within a bounded pilot window, you can produce:

* **Basis evidence:** per-window Polygon vs Kalshi expiration value distributions are stable and small relative to your quoting distance; “strike flip risk” is rare at your quote price distance. (`PLAN_OF_RECORD.md` §1.2; `tools/settlement_basis_audit.py`)
* **Fill evidence:** empirical maker fill curves show non-trivial fills at your intended quote style (or you learn it’s dead). (`PLAN_OF_RECORD.md` §2.4; `tools/build_fillcalib_dataset.py`)
* **Operational stability:** 24/7 supervisor runs for ≥1–2 weeks with no safety violations (no orders outside window, kill switch works, restarts reconcile). (`PLAN_OF_RECORD.md` §3; `exec/supervisor_index.py`, `sched/hotrestart.py`)

Pivot (or at least “pause pilot”) if:

* basis noise is large enough to routinely flip bin outcomes near where you must quote to get filled, **or**
* maker fills are near-zero without crossing (meaning maker-only is effectively “never trade”), **or**
* fees and required quote aggressiveness imply negative EV even with optimistic fill.

### “How far from making money?” (answered the only honest way)

* **What proof is missing?**

  * Measured maker fill curve by bin/price/time-to-expiry.
  * Basis audit showing settlement mapping risk is controlled.
  * Fee reconciliation (config matches observed realized fees).
  * A ledger with real fills (even tiny) and post-trade evaluation loop.
* **What minimum evidence would justify risking $X at 1‑lot scale?** (no guarantees)

  * **PAPER stability:** ≥5 consecutive market days of supervisor dry-runs with clean artifacts + no-go reasons that make sense.
  * **Basis audit:** for chosen series+window, basis std/quantiles are materially smaller than your strike spacing/typical quote distance; documented in a daily artifact.
  * **Pilot safety tests:** “live without ack fails,” “kill switch blocks submits,” “maker-only rejects crossing,” verified by tests + a staged AWS run.
  * **Fill dataset:** enough quote attempts to estimate fill probability with confidence bands (even if low); if fills are effectively 0, you shouldn’t risk $X at all—your “strategy” is just printing reports.

---

## 9. Prioritized next steps (1–2 weeks, 4–8 weeks)

### Next 1–2 weeks (credibility first: correctness, realism, instrumentation)

1. **Decouple index-only GO/NO‑GO from macro freshness**

   * Goal: index runs blocked only by index-relevant feeds/gates.
   * Evidence: go/no-go artifact shows scope, and index can GO while macro is stale. (`docs/PLAN_OF_RECORD.md` “Top blockers”; `exec/monitors/freshness.py` has scoping helpers)
2. **Make settlement basis audit a first-class gate + artifact**

   * Generate per-window audit for INX/INXU/NASDAQ100/NASDAQ100U; surface “flip risk.” (`tools/settlement_basis_audit.py`; `DATAFLOW.md`)
3. **Start TOB + quote-intent telemetry collection (bounded) on PAPER**

   * Log TOB snapshots and your intended quotes via `exec/collectors/tob_logger.py` and telemetry sink; build first conservative fill curves. (`PLAN_OF_RECORD.md` §2.4)
4. **Surface calibration ages + model metadata in one committed artifact**

   * Currently cited as missing. Make it impossible to “forget” stale calibration. (`KNOWN_ISSUES.md`; `exec/pilot_readiness.calibration_age_days`)
5. **Hard-enforce index-only scope at entrypoints**

   * Eliminate the “macro could run if misconfigured” foot-gun. (`KNOWN_ISSUES.md`; `utils/family.py`)
6. **Clock skew guardrails + AWS NTP verification**

   * Turn skew into a NO‑GO with clear remediation steps. (`KalshiClockSkewError`; `scan_ladders._clock_skew_seconds`)

### Next 4–8 weeks (controlled pilot readiness + measured execution)

1. **Run a tiny maker-only pilot (after Gate A exits) to measure fills**

   * Strict 1-lot, strict window guard, strict kill switch, and daily cap. (`docs/PLAN_OF_RECORD.md` §3 “Gate B — PILOT”)
2. **Calibrate fillratio/alpha + slippage models from real pilot data**

   * Populate `core/execution/index_models.py` curves from observed data, not assumptions.
3. **Replay parity + EV honesty gating in CI**

   * Prevent silent changes to EV math or proposal selection from creeping in. (`tools/replay.py`; `scripts/parity_gate.py`)
4. **Operationalize AWS: dashboards + alerts + crash recovery drills**

   * CloudWatch config exists; prove alerting, retention, and restart behavior in real runs. (`configs/cloudwatch/*`; runbooks)
5. **Post-trade evaluation loop (even if tiny size)**

   * Daily report: expected EV vs realized, fee breakdown, fill gaps, basis anomalies, and stop conditions.

---

## 10. HANDOFF FOR CODEX (machine-readable)

```yaml
top_priority_tasks:
  - goal: "Decouple index-only GO/NO-GO from macro freshness so index ladders can run independently."
    files_likely_touched:
      - "src/kalshi_alpha/exec/monitors/freshness.py"
      - "src/kalshi_alpha/core/gates/quality_gates.py"
      - "configs/freshness.index.yaml"
      - "configs/quality_gates.index.yaml"
      - "src/kalshi_alpha/exec/runners/scan_ladders.py"
      - "src/kalshi_alpha/exec/preflight_index.py"
    acceptance_criteria:
      - "When running index-only entrypoints, macro feeds do not cause NO-GO unless explicitly in index scope."
      - "go/no-go artifact includes an explicit 'scope' field and lists only scoped blockers."
      - "Existing pytest suite passes."
    recommended_tests_or_commands:
      - "pytest -q"
      - "make monitors"
      - "python -m kalshi_alpha.exec.preflight_index"
      - "python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run"

  - goal: "Promote settlement basis audit to a first-class gate with a daily artifact + strike flip risk summary."
    files_likely_touched:
      - "tools/settlement_basis_audit.py"
      - "src/kalshi_alpha/exec/preflight_index.py"
      - "src/kalshi_alpha/exec/monitors/runtime.py"
      - "configs/quality_gates.index.yaml"
      - "docs/PLAN_OF_RECORD.md"
    acceptance_criteria:
      - "New artifact written per series/day: basis summary + per-window deltas + 'flip risk' flags."
      - "Preflight or quality gate can fail-closed when basis audit missing/stale."
      - "Fixture-based test covers at least one synthetic window and validates output schema."
    recommended_tests_or_commands:
      - "pytest -q"
      - "python tools/settlement_basis_audit.py --help"
      - "python -m kalshi_alpha.exec.preflight_index"

  - goal: "Start bounded TOB + quote-intent telemetry capture for index ladders (PAPER-safe) and persist it durably."
    files_likely_touched:
      - "src/kalshi_alpha/exec/collectors/tob_logger.py"
      - "src/kalshi_alpha/exec/telemetry/sink.py"
      - "src/kalshi_alpha/exec/supervisor_index.py"
      - "src/kalshi_alpha/exec/runners/micro_index.py"
      - "configs/pilot.yaml"
    acceptance_criteria:
      - "During a dry-run window, TOB snapshots and quote intents are written with bounded depth/size."
      - "Telemetry includes run_id/window_id/market_ticker fields for correlation."
      - "Housekeeping prevents unbounded growth (retention policy documented)."
    recommended_tests_or_commands:
      - "pytest -q"
      - "make collect-polygon-ws"
      - "python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run"

  - goal: "Generate fill calibration dataset + conservative maker fill curves from telemetry; wire into fillprob/fillratio defaults."
    files_likely_touched:
      - "tools/build_fillcalib_dataset.py"
      - "src/kalshi_alpha/core/execution/fillprob.py"
      - "src/kalshi_alpha/core/execution/fillratio.py"
      - "src/kalshi_alpha/core/execution/index_models.py"
      - "src/kalshi_alpha/replay/fill_model.py"
    acceptance_criteria:
      - "Tool produces a dataset with sample counts and a derived fill curve per series/window bucket."
      - "Scanner can load fill curves and reports 'uncalibrated' only when no data exists."
      - "Conservative defaults used when sample size below threshold."
    recommended_tests_or_commands:
      - "pytest -q"
      - "python tools/build_fillcalib_dataset.py --help"
      - "make pilot-readiness"

  - goal: "Make calibration age visibility unavoidable: single summary artifact + scoreboard integration."
    files_likely_touched:
      - "src/kalshi_alpha/exec/pilot_readiness.py"
      - "src/kalshi_alpha/exec/scoreboard.py"
      - "jobs/calibrate_hourly.py"
      - "jobs/calibrate_close.py"
      - "docs/PLAN_OF_RECORD.md"
    acceptance_criteria:
      - "One artifact lists calibration ages per series/horizon and flags stale items."
      - "Scoreboard renders calibration age status for each series."
      - "NO-GO reasons explicitly include which calibration file(s) are stale/missing."
    recommended_tests_or_commands:
      - "pytest -q"
      - "make pilot-readiness"
      - "make calibrate-index"

  - goal: "Enforce index-only scope at production entrypoints to remove macro-footgun risk."
    files_likely_touched:
      - "src/kalshi_alpha/utils/family.py"
      - "src/kalshi_alpha/exec/pipelines/today.py"
      - "src/kalshi_alpha/exec/pipelines/daily.py"
      - "src/kalshi_alpha/exec/supervisor_index.py"
      - "docs/PLAN_OF_RECORD.md"
    acceptance_criteria:
      - "Default production runners refuse to run non-index families unless an explicit override is set."
      - "Unit test covers that FAMILY defaults to index or that non-index requires explicit flag."
    recommended_tests_or_commands:
      - "pytest -q"
      - "make paper_live_offline"

  - goal: "Harden broker boundary: fail-closed enforcement for pilot caps + maker-only + kill switch, independent of strategy logic."
    files_likely_touched:
      - "src/kalshi_alpha/brokers/kalshi/live.py"
      - "src/kalshi_alpha/brokers/kalshi/base.py"
      - "src/kalshi_alpha/exec/runners/scan_ladders.py"
      - "src/kalshi_alpha/exec/limits.py"
      - "configs/pilot.yaml"
    acceptance_criteria:
      - "Live broker refuses to submit without explicit acknowledgement + required env vars."
      - "Crossing orders are rejected in maker-only mode."
      - "Kill switch engaged causes no submits and emits a clear audit log event."
      - "Adds/updates tests referenced in TEST_COVERAGE.md (live safety)."
    recommended_tests_or_commands:
      - "pytest -q"
      - "make live-smoke  # read-only"
      - "python -m kalshi_alpha.exec.live_smoke"

  - goal: "Add explicit clock-skew guardrail and AWS NTP remediation path; surface in monitors + preflight."
    files_likely_touched:
      - "src/kalshi_alpha/brokers/kalshi/http_client.py"
      - "src/kalshi_alpha/exec/preflight_index.py"
      - "src/kalshi_alpha/exec/monitors/runtime.py"
      - "src/kalshi_alpha/exec/runners/scan_ladders.py"
      - "docs/runbooks/oncall_checks.md"
    acceptance_criteria:
      - "If skew exceeds threshold, system emits NO-GO with remediation instructions."
      - "Runtime monitor records skew metric and can alert."
      - "Includes regression test with injected clock behavior if supported."
    recommended_tests_or_commands:
      - "pytest -q"
      - "make monitors"

  - goal: "Operationalize replay parity + EV honesty as a CI gate to prevent silent EV drift."
    files_likely_touched:
      - "tools/replay.py"
      - "src/kalshi_alpha/core/archive/replay.py"
      - "scripts/parity_gate.py"
      - ".github/workflows/ci.yml"
      - "docs/DOCS_AND_LOGGING_SYSTEM.md"
    acceptance_criteria:
      - "CI runs parity gate on fixtures and fails if EV outputs drift beyond tolerance."
      - "Artifacts include a diff summary (what changed and why)."
    recommended_tests_or_commands:
      - "pytest -q"
      - "make parity-ci"
      - "python tools/replay.py --help"

  - goal: "Finalize AWS 24/7 wiring: systemd units, CloudWatch logs/metrics, and a minimal on-call alert surface."
    files_likely_touched:
      - "configs/systemd/kalshi-*.service"
      - "configs/systemd/kalshi-*.timer"
      - "configs/cloudwatch/kalshi-supervisor-index.json"
      - "docs/runbooks/aws_supervisor_index.md"
      - "src/kalshi_alpha/exec/supervisor_index.py"
    acceptance_criteria:
      - "Supervisor runs continuously with auto-restart and writes heartbeats/monitor artifacts."
      - "CloudWatch receives logs + at least heartbeat/freshness metrics."
      - "Documented crash recovery drill: restart supervisor and reconcile outstanding orders safely."
    recommended_tests_or_commands:
      - "make aws-deploy-dashboards"
      - "python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run"
      - "make monitors"
```
