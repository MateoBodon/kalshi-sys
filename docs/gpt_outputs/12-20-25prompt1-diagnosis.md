## 1. Project snapshot (5–10 bullets)

* **Date / ground truth used:** 2025-12-20; primary ground truth is the provided `project_state` snapshot (all files in `kalshi_project_state_12-20-25/*`).
* **Docs consulted (outline):** `PIPELINE_FLOW.md` (end‑to‑end scan → propose → gate → (optional) execute), `DATAFLOW.md` (raw/proc/reference layouts + artifacts), `ARCHITECTURE.md` (system components + 24/7 supervisor), `MODULE_SUMMARIES.md` + `FUNCTION_INDEX.md` (where the actual code lives), `CURRENT_RESULTS.md` + `EXPERIMENTS.md` (what’s been run / what evidence exists), `KNOWN_ISSUES.md` + `OPEN_QUESTIONS.md` (risk + missing proof), `CONFIG_REFERENCE.md` + `SERVER_ENVIRONMENT.md` + `TEST_COVERAGE.md` + `CHANGELOG.md`.
* **Primary implemented target:** an index-ladder pipeline for **INX / NASDAQ100** “close” ladders + **INXU / NASDAQ100U** hourly (“U”) ladders, with a “maker‑first, EV‑after‑fees” thesis (`ARCHITECTURE.md`, `PIPELINE_FLOW.md`, `RESEARCH_NOTES.md`).
* **Core execution surface exists:** dry broker + live broker adapters are present (`src/kalshi_alpha/brokers/kalshi/dry.py`, `src/kalshi_alpha/brokers/kalshi/live.py` per `MODULE_SUMMARIES.md`), and the main scanner/runner is `src/kalshi_alpha/exec/runners/scan_ladders.py` (also per `MODULE_SUMMARIES.md`).
* **But: evidence is currently “NO-GO.”** Committed artifacts show **no ledger fills** and pilot readiness is **NO‑GO due to insufficient_data** for all four index series (`CURRENT_RESULTS.md`, `KNOWN_ISSUES.md`).
* **Operational gating exists, but it’s currently blocking:** GO/NO‑GO artifacts are false due to stale feeds / stale monitors / stale heartbeat (`CURRENT_RESULTS.md`, `KNOWN_ISSUES.md`).
* **Key research infrastructure exists:** calibration jobs for sigma/time‑of‑day and drift curves (`jobs/calib_hourly.py`, `jobs/calib_eod.py`, `jobs/_index_calibration.py`), replay utilities (`src/kalshi_alpha/replay/polygon_index_replay.py`), and “honesty” / reliability scoring hooks (`report/honesty.py` referenced in `RESEARCH_NOTES.md`).
* **Biggest “today” reality check:** the repo is instrumented to *propose* and *paper-simulate*, but **it has not demonstrated positive EV after fees with realistic fills** in the committed evidence set (explicitly: “fills = 0”, “sample size = 0”) (`CURRENT_RESULTS.md`).

---

## 2. System framing (what we’re trading + edge hypothesis)

### Instruments: what exactly are we trading?

* **FACT (contract terms):**

  * $INX contract underlying is “the price of the S&P 500 Index `<on/before> <time> on <date>`,” and only times inside **9:30 AM–4:00 PM ET** are listed; if no data exists at `<time>`, the expiration value uses the most recent value before that time. 
  * Nasdaq-100 contract terms are structurally the same: underlying is “price of the Nasdaq‑100 Index `<on/before> <time> on <date>`,” with the same market-hours restriction and “last available value” fallback when no data exists at the time. ([Commodity Futures Trading Commission][1])
  * Both state **Source Agency: Kalshi**. 
* **Implication (risk officer view):** “Source Agency is Kalshi” means your Polygon feed is **not** automatically the settlement truth. There is **basis risk** between Polygon’s index print and whatever Kalshi documents at expiration time.

### Cadence / decision moments (index ladders focus)

* **FACT (repo intent):** system is built around:

  * **Hourly intraday** windows for “U” series (INXU/NASDAQ100U), and
  * **Close** windows for INX/NASDAQ100 (daily close ladders).
    This is explicit throughout `PIPELINE_FLOW.md` and the index scanner modules in `MODULE_SUMMARIES.md` (e.g., `src/kalshi_alpha/exec/scanners/scan_index_hourly.py`, `scan_index_close.py`, `exec/supervisor_index.py`).

### Fee model constraints (what “EV after fees” means here)

* **FACT (official schedule):** Kalshi’s fee schedule includes a **special index fee formula** for INX and NASDAQ100: fee is `round up(0.035 × C × P × (1 − P))` to the nearest cent. ([Kalshi][2])
* **ASSUMPTION (must be verified in-system):** whether **maker** fees are truly 0 for these index ladders (your repo mentions/guards index maker behavior via `kalshi_alpha.core.fees._ensure_index_maker_fee_guard()` per `MODULE_SUMMARIES.md`, but I’m not treating “maker=0” as a verified external fact without an official line item explicitly stating that).

### Edge hypothesis (what is supposed to create EV after fees?)

* **FACT (repo framing):** “Estimate a ladder PMF/CDF for the index at target time, align to strike grid, compute EV after fees, and execute maker‑first when EV > 0 subject to risk gates.” (`RESEARCH_NOTES.md`, `PIPELINE_FLOW.md`, `ARCHITECTURE.md`).
* **Mechanically, the intended edge is:**

  * **Probability edge:** your PMF better approximates true probability than market-implied odds (after calibration).

    * Core primitives: `kalshi_alpha.models.pmf_index` (sigma_tod curves + drift), `kalshi_alpha.core.pricing.align` (CDF→strike projection), and index strategy modules (`kalshi_alpha.strategies.index.*`) per `RESEARCH_NOTES.md` + `MODULE_SUMMARIES.md`.
  * **Microstructure edge (maker-first):** you capture spread / rebates / favorable queue fills vs paying taker fees (requires fill realism).

    * Repo has microprice signal + replacement throttling (`src/kalshi_alpha/exec/quote_microprice.py`, `quote_optim.py`) and fill/slippage models (`core/execution/fillprob.py`, `fillratio.py`, `slippage.py`) per `MODULE_SUMMARIES.md`.
  * **Operational edge:** you stay live and correct (fresh data, correct windows, correct strikes, safe execution) when others are not; this is gated via GO/NO‑GO + preflight + kill switch (see below).

### Required realities for profitability (non-negotiables)

These are the things that must be true for “EV after fees” to mean anything:

* **Fee correctness:** you must apply the correct fee schedule (incl. rounding) to the side you actually take. (External formula exists. ([Kalshi][2]) Repo implements fee loading in `src/kalshi_alpha/core/fees/*` per `MODULE_SUMMARIES.md`.)
* **Fill realism:** maker fill probability + queue position + cancel/replace throttles must be modeled or measured. Right now this is explicitly a gap (“not calibrated”) (`KNOWN_ISSUES.md`).
* **Latency bounds:** if you quote, you must update before you get run over; otherwise “edge” becomes adverse selection. Repo has WS freshness gating and window guards (`exec/supervisor_index.py`, `exec/window_guard.py` per `MODULE_SUMMARIES.md`), but the evidence of live performance isn’t in committed artifacts.
* **Calibration stability:** sigma_tod / drift curves must remain valid across regimes; repo has PIT bias and calibration jobs, but you need live drift alarms + stability reports (`jobs/_index_calibration.py`, `report/honesty.py` referenced in `RESEARCH_NOTES.md`).
* **Risk caps enforced in execution:** PAL/VAR/drawdown must block runaway exposure; repo has these pieces (`core/risk/*`, `risk/var_index.py`, `risk/correlation.py`, `configs/pal_policy.yaml`, `configs/index_var.yaml`, `configs/index_correlation.yaml` in `CONFIG_REFERENCE.md`).

---

## 3. Pipeline reconstruction (end-to-end)

This is the repo’s actual “scan → price → decide → (optional) place orders → monitor/report” as reconstructed from `PIPELINE_FLOW.md`, `DATAFLOW.md`, and code pointers in `MODULE_SUMMARIES.md` / `FUNCTION_INDEX.md`.

### Stage 0 — Config + secrets + environment

* **Key files/modules**

  * `SERVER_ENVIRONMENT.md` (env vars; data roots; dependencies)
  * Config loaders: `src/kalshi_alpha/config/index_ops.py`, `kalshi_alpha.config.size_ladder`, `kalshi_alpha.core.gates.*` (all referenced in `CONFIG_REFERENCE.md`)
* **Key invariants**

  * `POLYGON_API_KEY` available; Kalshi auth present if live (`KALSHI_API_KEY_ID`, `KALSHI_PRIVATE_KEY_PEM_PATH`, `KALSHI_ENV`) (`SERVER_ENVIRONMENT.md`).
  * Data directories exist: `data/raw`, `data/proc`, `data/reference`; runtime state under `data/proc/state` (`SERVER_ENVIRONMENT.md`, `DATAFLOW.md`).
* **Likely failure modes**

  * Missing env vars → silent fallback to dry mode or partial failure (preflight exists for index windows: `src/kalshi_alpha/exec/preflight_index.py` in `MODULE_SUMMARIES.md`).
  * Wrong timezone / DST drift (repo has tz_not_et monitor in quality gates: `configs/quality_gates.index.yaml` in `CONFIG_REFERENCE.md`).
* **Tested vs untested**

  * Some auth/signing logic is tested (per `TEST_COVERAGE.md`), but end-to-end env gating is typically untested.

### Stage 1 — Data ingest (Polygon indices; optionally replay)

* **Key files/modules**

  * Raw ingest: `src/kalshi_alpha/exec/ingest/polygon_index.py` (history downloader) (`MODULE_SUMMARIES.md`)
  * WS listening/replay: `drivers/polygon_index_ws.py` and `src/kalshi_alpha/replay/polygon_index_replay.py` (`ARCHITECTURE.md`, `MODULE_SUMMARIES.md`)
  * Data layout: `DATAFLOW.md` (“Storage layout”).
* **Invariants**

  * WS freshness must be within configured thresholds (index preflight + freshness monitors; see `exec/preflight_index.py`, `exec/monitors/freshness` referenced in `CONFIG_REFERENCE.md` and `MODULE_SUMMARIES.md`).
  * Index “snapshot” at decision time is correctly timestamped and ET-aligned.
* **Failure modes**

  * Stale WS feed (explicitly appears in artifacts as “Polygon WS age …” in scoreboards; but currently GO/NO‑GO is also tripping on *macro feeds* unrelated to index trading) (`CURRENT_RESULTS.md`, `OPEN_QUESTIONS.md`).
  * Replay divergence: replayed WS messages not matching live pipeline (needs parity tests).
* **Tested vs untested**

  * There are smoke tests with fixtures for scanners (`TEST_COVERAGE.md`), but ingest reliability and WS reconnect/backoff behavior is usually lightly tested.

### Stage 2 — Market discovery (Kalshi markets → strikes/bins)

* **Key files/modules**

  * `src/kalshi_alpha/markets/discovery.py`: discover markets for day and group by scheduler windows (`MODULE_SUMMARIES.md`)
  * Kalshi public client: `kalshi_alpha.core.kalshi_api.KalshiPublicClient` referenced in `FUNCTION_INDEX.md` via pipelines and runners.
* **Invariants**

  * Correct mapping from discovered market tickers → series family (INX/INXU/NASDAQ100/NASDAQ100U).
  * Correct extraction of ladder strikes and event timestamp labels.
* **Failure modes**

  * Wrong window grouping (e.g., mixing noon/close; or wrong ET conversion).
  * Strike grid changes intraday, causing “off-by-one rung” misalignment (this is the kind of bug that silently kills EV; repo has monotonicity checks and prob mass checks in pricing modules, but you still need explicit regression tests on strike extraction).
* **Tested vs untested**

  * Discovery logic is typically lightly tested; not highlighted as covered in `TEST_COVERAGE.md`.

### Stage 3 — Model + calibration → PMF/CDF on strike grid

* **Key files/modules**

  * Calibration jobs: `jobs/calib_hourly.py`, `jobs/calib_eod.py`, shared `jobs/_index_calibration.py` (`MODULE_SUMMARIES.md`)
  * PMF loader/model: `src/kalshi_alpha/models/pmf_index.py` (`MODULE_SUMMARIES.md`)
  * PMF alignment: `src/kalshi_alpha/core/pricing/align.py` (piecewise CDF projection) (`MODULE_SUMMARIES.md`)
  * Index strategy PMF: `src/kalshi_alpha/strategies/index/*` (e.g., `model_polygon.py`, `cdf.py`) (`RESEARCH_NOTES.md`, `MODULE_SUMMARIES.md`)
* **Invariants**

  * Calibration files exist and are recent enough (preflight checks file age: `_file_age_days` in `exec/preflight_index.py` per `MODULE_SUMMARIES.md`).
  * PMF sums to 1 (or is simplex-projected without distortion).
  * Target time mapping is consistent with contract terms (“traditional market hours 9:30–4 ET”) 
* **Failure modes**

  * **Model oversimplification:** sqrt(time) Student‑t/normal assumptions without conditioning on realized vol or intraday microstructure (explicit limitation) (`KNOWN_ISSUES.md`).
  * **Calibration drift:** sigma_tod curves stale; PIT bias changes; leads to systematic mispricing.
* **Tested vs untested**

  * Core alignment is unit-tested (`TEST_COVERAGE.md` mentions `core/pricing/align` tests).
  * End‑to‑end calibration validity and drift detection are not clearly evidenced in committed artifacts (`CURRENT_RESULTS.md`).

### Stage 4 — Market reconciliation → EV after fees (+ fill/slippage penalties)

* **Key files/modules**

  * Scanner helpers:

    * `src/kalshi_alpha/exec/scanners/scan_index_hourly.py` (hourly)
    * `src/kalshi_alpha/exec/scanners/scan_index_close.py` (close)
    * common: `src/kalshi_alpha/exec/scanners/index_scan_common.py`
    * EV utilities: `src/kalshi_alpha/exec/scanners/utils.py`
      (all in `MODULE_SUMMARIES.md`)
  * Fees: `src/kalshi_alpha/core/fees/__init__.py`, `core/fees/index_series.py` (`MODULE_SUMMARIES.md`)
  * Fill/slippage: `core/execution/fillprob.py`, `fillratio.py`, `slippage.py`, `exec/quote_microprice.py`, `exec/quote_optim.py` (`MODULE_SUMMARIES.md`)
* **Invariants**

  * EV uses correct fee rounding semantics (repo has `round_up_to_cent` in fee schedule utilities).
  * Any “maker” assumption about fills is **penalized** by realistic fill prob (or you don’t trade).
* **Failure modes**

  * **Optimistic fills:** treating top-of-book as free liquidity; ignoring queue priority; ignoring cancel/replace delays (explicitly listed as “fill model calibration gap”) (`KNOWN_ISSUES.md`).
  * Mispricing artifacts: implied curve non-monotonicity; PMF mass gaps; repo has mispricing analytics (`core/pricing/mispricing.py`) but enforcement depends on gates.
* **Tested vs untested**

  * Some pricing primitives are tested, but execution realism (fill/slippage) is not credibly validated without real fills (`CURRENT_RESULTS.md`).

### Stage 5 — Opportunity selection + risk/quality gating

* **Key files/modules**

  * Primary runner: `src/kalshi_alpha/exec/runners/scan_ladders.py` (73+ functions; proposal generation; EV honesty gate; execution hooks) (`MODULE_SUMMARIES.md`)
  * Quality gates: `src/kalshi_alpha/core/gates/quality_gates.py` (run_quality_gates) (`MODULE_SUMMARIES.md`, `FUNCTION_INDEX.md`)
  * Index preflight: `src/kalshi_alpha/exec/preflight_index.py` (`MODULE_SUMMARIES.md`)
  * Risk: `src/kalshi_alpha/core/risk/*`, `risk/var_index.py`, `risk/correlation.py`, sizing in `core/sizing/kelly.py` (`MODULE_SUMMARIES.md`)
  * Pilot constraints: `configs/pilot.yaml`, `configs/size_ladder.yaml`, `configs/pal_policy.yaml` (`CONFIG_REFERENCE.md`)
* **Invariants**

  * GO/NO‑GO must block execution (and ideally block even proposal writing if you’re “NO-GO” for that window).
  * PAL/drawdown caps must be enforced before any live order is emitted.
  * Pilot restrictions must be enforced in “pilot mode.”
* **Failure modes**

  * **Gate coupling:** index trading blocked because unrelated macro feeds are stale (explicit in artifacts and open questions) (`CURRENT_RESULTS.md`, `OPEN_QUESTIONS.md`, `KNOWN_ISSUES.md`).
  * Misconfigured window guard → trading outside intended window.
* **Tested vs untested**

  * Gate config is present, but gate behavior under degraded feeds needs explicit tests and clear policy (index-only vs global) (`OPEN_QUESTIONS.md`).

### Stage 6 — Execution (dry vs live) + state tracking

* **Key files/modules**

  * Brokers:

    * Factory: `src/kalshi_alpha/brokers/__init__.py` (create_broker)
    * Dry broker: `src/kalshi_alpha/brokers/kalshi/dry.py`
    * Live broker: `src/kalshi_alpha/brokers/kalshi/live.py`
    * Auth/retry/rate limiting: `brokers/kalshi/http_client.py` and `ws_client.py`
      (all in `MODULE_SUMMARIES.md`)
  * Order queue: `src/kalshi_alpha/core/execution/order_queue.py`
  * Outstanding orders state: `src/kalshi_alpha/exec/state/orders.py`
  * Paper ledger: `src/kalshi_alpha/exec/ledger/__init__.py`
* **Invariants**

  * Live mode should require explicit, hard-to-misuse gates (`configs/pilot.yaml` includes `pilot.require_live_broker`, `pilot.require_acknowledgement` in `CONFIG_REFERENCE.md`).
  * Kill switch must abort order sends.
* **Failure modes**

  * Partial fills not reconciled; outstanding state diverges from exchange.
  * Cancel/replace spam trips rate limits; stale quotes linger.
* **Tested vs untested**

  * HTTP signing/retry likely tested; live end-to-end cancel/replace and reconciliation is not evidenced in `TEST_COVERAGE.md`.

### Stage 7 — Monitoring, reporting, replay, post-trade evaluation

* **Key files/modules**

  * Reports: `src/kalshi_alpha/exec/reports/__init__.py` (markdown report), `exec/reports/ramp.py` (pilot readiness) (`MODULE_SUMMARIES.md`)
  * Monitors: `src/kalshi_alpha/exec/monitors/runtime.py` (kill switch / drawdown / WS disconnects / sequential guard etc — see `FUNCTION_INDEX.md` excerpt around `compute_runtime_monitors`)
  * Scoreboards: `src/kalshi_alpha/exec/scoreboard.py`, `exec/scoreboard_index_paper.py`
  * Telemetry: `src/kalshi_alpha/exec/telemetry/sink.py`, shipper `exec/telemetry/shipper.py`
  * SLO + CloudWatch: `src/kalshi_alpha/exec/slo.py` (CloudWatch push) (`MODULE_SUMMARIES.md`)
  * Replay/parity: `src/kalshi_alpha/replay/*`, `tools/replay.py`, `scripts/parity_gate.py` (`RESEARCH_NOTES.md`, `MODULE_SUMMARIES.md`)
* **Invariants**

  * Scoreboards must reflect reality: expected vs realized EV, fills, slippage, freshness, time-at-risk.
* **Failure modes**

  * If ledger is empty, scoreboards give false comfort (“freshness ok”) while strategy has zero empirical proof (this is exactly the current state) (`CURRENT_RESULTS.md`).
* **Tested vs untested**

  * Reporting is generally untested; telemetry formats might be stable but not validated.

---

## 4. Experimental evidence (what’s been run + how trustworthy)

### What has ACTUALLY been run (from repo artifacts + docs)

* **FACT:** `EXPERIMENTS.md` lists runnable components:

  * Calibration jobs (`jobs/calib_hourly.py`, `jobs/calib_eod.py`)
  * Scanners (`exec/scanners/scan_index_hourly.py`, `scan_index_close.py`)
  * Replay (`replay/polygon_index_replay.py`)
  * Execution realism tuning (`core/execution/fillratio.tune_alpha`, `core/execution/slippage.fit_slippage`)
  * Scoreboards (`exec/scoreboard.py`)
  * Supervisor (`exec/supervisor_index.py`)
    (`EXPERIMENTS.md`, `MODULE_SUMMARIES.md`)
* **FACT:** committed results show:

  * Scoreboards: “freshness OK,” but **no ledger data available** (`CURRENT_RESULTS.md`)
  * Pilot readiness: **NO‑GO** for all four index series due to **insufficient_data (fills=0)** (`CURRENT_RESULTS.md`)
  * GO/NO‑GO: `go=false` with stale feed + stale heartbeat/monitor reasons (`CURRENT_RESULTS.md`)

### Any “EV/alpha” claims — are fills/slippage/fees realistic?

* **FACT:** Fees are modeled (fee schedule JSON → `kalshi_alpha.core.fees.*`; plus `dev/parse_fees.py`) (`MODULE_SUMMARIES.md`).
* **FACT:** Fill and slippage machinery exists (`core/execution/fillprob.py`, `fillratio.py`, `slippage.py`), but the repo itself states **fill model calibration is a gap** (`KNOWN_ISSUES.md`).
* **Bottom line:** **No committed evidence** demonstrates positive EV after fees with realistic fills, because **there are zero fills in the committed ledger artifacts** (`CURRENT_RESULTS.md`, `KNOWN_ISSUES.md`).

### Top 5 ways a backtest/paper run can be accidentally optimistic (in this repo’s current state)

1. **Queue/fill fantasy:** assuming maker orders fill at your posted price without modeling queue position, trade-through, cancel latency (`KNOWN_ISSUES.md`; fill calibration gap).
2. **Settlement basis mismatch:** using Polygon’s index print as “truth” while contract terms state the expiration value is what **Kalshi documents** (Source Agency: Kalshi). 
3. **Hindsight leakage via strike grid / event filtering:** discovery could select “nice” strikes or only windows where data exists; must test discovery determinism and missing-data behavior. (This is implied risk in `markets/discovery.py` + “if no data available use last value” contract terms.) 
4. **Fee side confusion:** applying index fee formula but not correctly mapping “maker vs taker” or the actual trade side you end up with (especially if you quote but end up crossing). ([Kalshi][2])
5. **Gate‑masked drift:** GO/NO‑GO currently fails due to stale macro feeds; if you “force-run” anyway, you could be testing an un-gated pipeline and inadvertently excluding bad regimes in recorded artifacts (`CURRENT_RESULTS.md`, `OPEN_QUESTIONS.md`).

### 3–6 targeted debug experiments to validate realism fast

1. **Settlement basis audit (must-do):** build a daily job that compares **Polygon index values at each expiration timestamp** vs the **value Kalshi uses/records** for that window. Flag basis distribution + tail events.

   * Justification: contract terms explicitly make Kalshi the source agency. 
2. **TOB snapshot → fill curve calibration:** start collecting top-of-book snapshots for the exact ladders you quote, then run `src/kalshi_alpha/replay/fill_model.py` to build conservative fill probability curves and feed them into `core/execution/fillprob.py`. (`RESEARCH_NOTES.md`, `MODULE_SUMMARIES.md`)
3. **Cancel/replace stress test in paper + telemetry:** use the existing `ReplacementThrottle` (`exec/quote_microprice.py`) + `OrderQueue` (`core/execution/order_queue.py`) to simulate high-churn conditions; measure “time stale” and “quotes live near expiry.”
4. **EV honesty parity gate on index-only:** run the pipeline with macro feeds disabled/ignored and confirm the “EV honesty” machinery (`exec/reports/ramp.py`, `report/honesty.py` per `RESEARCH_NOTES.md`) produces stable calibration metrics for INX/NDX windows.
5. **Maker vs taker outcome attribution:** instrument every fill (paper or live) as maker/taker, record realized fee per fill from the fee schedule module, and verify it matches the official fee formula for indices. ([Kalshi][2])

---

## 5. Execution & risk controls (paper/pilot/live readiness)

### What gates prevent accidental live trading?

* **FACT (repo config surface):** `configs/pilot.yaml` contains explicit pilot constraints:

  * allowed series, maker-only enforcement, max contracts/order, max daily/weekly loss, require acknowledgement, require live broker, session prefix (`CONFIG_REFERENCE.md` → `configs/pilot.yaml`).
* **FACT (repo kill switch):** kill switch functions exist (`src/kalshi_alpha/exec/heartbeat.py`: `kill_switch_engaged`, `resolve_kill_switch_path`, etc.) per `FUNCTION_INDEX.md`.
* **FACT (preflight):** index preflight checks exist (`src/kalshi_alpha/exec/preflight_index.py`: calibration file age, polygon ping, missing env vars) per `MODULE_SUMMARIES.md`.
* **FACT (risk guards):**

  * PAL policy + max loss per order includes fees (`core/risk/__init__.py:max_loss_for_order`) (`MODULE_SUMMARIES.md`)
  * Drawdown caps persist state (`core/risk/drawdown.py`) (`MODULE_SUMMARIES.md`)
  * VaR limiters exist (`risk/var_index.py`, `risk/correlation.py`) (`MODULE_SUMMARIES.md`, `CONFIG_REFERENCE.md`)

### Are these sufficient for a tiny pilot (e.g., 1-lot maker-only)?

* **Mostly yes in *design*, not yet yes in *validated reality*.**
  The presence of PAL + pilot config + kill switch + replacement throttling is good structure. The missing piece is **proving they’re enforced on the exact path that sends orders** (`scan_ladders.py.execute_broker` → broker adapter).
* **Specific “pilot foot-guns” I would block before any real orders:**

  * **Order reconciliation:** I see `OutstandingOrdersState` (`exec/state/orders.py`) and `OrderQueue` for cancel/replace, but I don’t see evidence (in artifacts) that restart recovery/cancel-all is enforced. This is a classic “wake up short vol” failure mode.
  * **Maker-only enforcement at the broker boundary:** `pilot.enforce_maker_only` exists, but it must be enforced **right before order submission** and on replacements too. (This likely lives in `scan_ladders.py._enforce_broker_guards()` + broker adapter; verify wiring.)
  * **Clock skew:** there’s a `KalshiClockSkewError` in HTTP client and also clock skew logic in `scan_ladders.py` (function list includes `_clock_skew_seconds`); ensure skew blocks execution not just logs.
  * **Kill switch semantics:** ensure kill switch is checked **before** any network call, and also stops replacements/cancels (not just new orders).

### Paper vs pilot vs live — where are we really?

* **FACT:** current committed evidence is still effectively **paper/dry**: scoreboards say **no ledger data** and pilot readiness has **fills=0** (`CURRENT_RESULTS.md`).
* **ASSUMPTION:** the live broker exists and likely works at the request-signing layer, but without logged fills you haven’t validated end-to-end exchange interaction, reconciliation, or real-world fill quality.

---

## 6. AWS / ops readiness (audit)

### What’s already there (good signs)

* **Supervisor component exists:** `src/kalshi_alpha/exec/supervisor_index.py` is explicitly “24/7 supervisor … orchestrates live index ladder scans,” with WS freshness gating (via `WSListener`) (`MODULE_SUMMARIES.md`, `ARCHITECTURE.md`).
* **Telemetry + metrics hooks exist:**

  * Telemetry sink (`exec/telemetry/sink.py`) writes durable JSONL with daily rotation (`MODULE_SUMMARIES.md`).
  * SLO exporter can publish to CloudWatch (`exec/slo.py`) (“best effort”) (`MODULE_SUMMARIES.md`).
* **Deployment templates exist (at least by reference):** `configs/systemd/*` and `configs/logrotate/*` are called out in `CONFIG_REFERENCE.md`, implying planned process supervision + log rotation.
* **Container build path exists:** `docker/aws-jobs/Dockerfile` is referenced in `SERVER_ENVIRONMENT.md`.

### Ops foot-guns / gaps (what makes 24/7 fragile today)

* **FACT:** “AWS wiring gap for index supervisor” is explicitly listed (`KNOWN_ISSUES.md`).
* **Policy coupling issue:** GO/NO‑GO currently fails due to stale macro feeds even though index pipeline could be the only active family (`OPEN_QUESTIONS.md`, `KNOWN_ISSUES.md`). That’s an ops problem: you’ll get false NO‑GO and stop trading unintentionally.
* **Secrets handling is not fully specified:** env vars are documented (`SERVER_ENVIRONMENT.md`), but I don’t see (in the snapshot) an explicit SSM/Secrets Manager policy, rotation, or blast-radius plan.
* **Crash recovery / idempotency not proven:** outstanding orders state exists, but there’s no committed evidence that restart recovery cancels or reconciles correctly.
* **Replayability is good in theory, but not in ops loop:** replay tools exist; what’s missing is a routine “last N windows replay parity” job that runs automatically and pages you when parity breaks.

---

## 7. Biggest gaps, failure modes, and realism risks

Ranked roughly by “could lose money / waste months” severity:

1. **Zero empirical trading evidence for index ladders right now.** You can’t claim EV or even directional correctness because **fills=0** and scoreboards have no ledger (`CURRENT_RESULTS.md`).
2. **Settlement basis risk is untreated.** Contract terms explicitly anchor expiration value to what **Kalshi** documents, not Polygon. If Polygon differs by even small amounts near strike boundaries, you’ll “model the wrong underlying.” 
3. **Fill model not calibrated = edge likely illusory.** You have fill/slippage modules, but the repo itself flags the calibration gap (`KNOWN_ISSUES.md`). In maker-first strategies, this is where most “paper alpha” dies.
4. **GO/NO‑GO gating is currently noisy / mis-scoped.** Index trading appears blocked by stale macro feeds; that’s both an availability issue and a research confound (`CURRENT_RESULTS.md`, `OPEN_QUESTIONS.md`).
5. **Fees can erase tiny edges.** With the index fee formula rounded up to cents, small mispricings are easily consumed by fees/spread unless you truly capture maker economics. ([Kalshi][2])
6. **Order lifecycle risk:** without strong reconciliation + cancel-all, you can accumulate stale inventory or leave resting orders into regime shifts. You have components (`OrderQueue`, `OutstandingOrdersState`), but not validated.
7. **Time window / DST hazards:** contract terms restrict times and are ET-based; your pipeline must be perfect on ET conversion and “target label” parsing. 
8. **Model simplification risk:** sqrt(time) + Student‑t/normal with light conditioning is fragile in volatility spikes (`KNOWN_ISSUES.md`).
9. **Liquidity assumptions:** Kalshi advertises market maker coverage in INX/NDX hourly increments during stock market days (good for spreads), but it also means you’re playing against professional quoting most of the time. ([Kalshi Help Center][3])
10. **Test coverage gaps around “the scary parts”:** unit tests cover some math; but execution, discovery, and ops are where losses happen (`TEST_COVERAGE.md`, `KNOWN_ISSUES.md`).

---

## 8. Continue / pivot assessment (with decision criteria)

### Blunt assessment

* **Continue, but narrow and “prove it” before scaling.**
  The architecture is coherent and the right components exist for a controlled pipeline (`PIPELINE_FLOW.md`, `ARCHITECTURE.md`). But **you are not in a position to claim edge** or responsibly run real capital until you produce **real fill + settlement + fee + slippage evidence**.

### Decision criteria (what proof is missing?)

Minimum proof missing today (based on `CURRENT_RESULTS.md` showing zero fills):

* **Settlement mapping proof:** quantified basis between Polygon index values and the value Kalshi uses at expiration. (No proof shown today.) 
* **Fill realism proof:** empirical maker fill rates vs queue/latency, not just heuristics (`KNOWN_ISSUES.md`).
* **EV tracking proof:** expected EV after fees vs realized PnL on actual fills (not “paper fills”).
* **Ops proof:** 2+ weeks of uninterrupted runs in the intended windows with correct GO/NO‑GO, heartbeat, and kill switch behavior (right now the go/no-go artifact is false) (`CURRENT_RESULTS.md`).

### “What minimum evidence would justify risking $X at 1-lot scale?”

For *any* nontrivial $X (even “small pilot money”), I’d require:

* **A controlled pilot mode** that is maker-only, 1-lot capped, and hard-gated by acknowledgement + kill switch + explicit allowed series (you have the config surface; enforce it at the broker boundary) (`CONFIG_REFERENCE.md`, `FUNCTION_INDEX.md`).
* **At least dozens to low-hundreds of *real* fills** with:

  * logged maker/taker classification,
  * realized fees computed from your fee module vs the official formula,
  * and realized outcomes matched to Kalshi expiration values. ([Kalshi][2])
* **Evidence that your realized edge survives pessimistic assumptions:** e.g., you still look positive if you haircut fills (lower fill prob) and widen slippage. (This is not something I can assert exists today; it’s a required test.)

Pivot triggers (when I would say “stop doing this”):

* If settlement basis error is large enough to flip bin outcomes around common strikes, **Polygon-only** settlement modeling is not viable; you’d need a direct “Kalshi source” capture or a robust basis model.
* If maker fills are too rare (or too adverse-selected near expiry), the maker-first approach may not be viable; you may need to pivot to a different execution style (still within index ladders) or accept that edge is not present.

---

## 9. Prioritized next steps (1–2 weeks, 4–8 weeks)

### Next 1–2 weeks (fastest credibility: correctness + realism + instrumentation)

1. **Fix GO/NO‑GO scope for index-only trading**

   * Make index runs depend on index-relevant freshness + calibration, not stale macro feeds (`OPEN_QUESTIONS.md`, `KNOWN_ISSUES.md`, `CONFIG_REFERENCE.md` → `quality_gates.index.yaml`).
2. **Build the settlement basis audit**

   * Quantify Polygon vs Kalshi expiration value at the exact `<time>` on `<date>` for INX/NDX windows. This is foundational because Source Agency is Kalshi. 
3. **Start collecting TOB snapshots and trade prints for ladders you quote**

   * Use existing telemetry sink (`exec/telemetry/sink.py`) or add a dedicated snapshot logger; then run `replay/fill_model.py` to build conservative fill curves.
4. **Enforce pilot safety at the last mile**

   * Ensure `pilot.require_acknowledgement`, `pilot.require_live_broker`, `pilot.enforce_maker_only`, size caps, and kill switch are enforced right before broker submissions (`scan_ladders.py.execute_broker`, `brokers/kalshi/live.py`, `exec/heartbeat.py`).
5. **Regenerate scoreboards with fresh data**

   * Right now scoreboards are stale/empty; make the pipeline produce fresh artifacts and show fill=0 explicitly if still paper-only (`CURRENT_RESULTS.md`, `ROADMAP.md`).

### Next 4–8 weeks (controlled pilot → monitored live)

1. **Run a tiny “maker-only 1-lot” pilot on one series first**

   * Pick **one** of INXU or NASDAQ100U; don’t split attention. Use `configs/size_ladder.yaml` stage A and `configs/pilot.yaml` to hard-cap risk.
2. **Implement robust order reconciliation**

   * On startup and periodically: fetch open orders, reconcile with `OutstandingOrdersState`, cancel stale ones, and enforce “no orders outside windows.”
3. **Calibrate fill and slippage models from real outcomes**

   * Use `fillratio.tune_alpha()` and `slippage.fit_slippage()` **only** once you have real fill/slippage data; persist curves and surface sample size in scoreboards.
4. **Add regression tests for discovery + strike alignment**

   * Freeze a set of known tickers/windows and ensure strike extraction + PMF alignment are stable; add tests for DST boundary weeks.
5. **AWS hardening**

   * Wire `supervisor_index` into AWS scheduling/supervision (systemd or ECS), add alarms on heartbeat stale, WS disconnect rate, and “orders live outside window.” (`KNOWN_ISSUES.md`, `CONFIG_REFERENCE.md`, `SERVER_ENVIRONMENT.md`)

---

## 10. HANDOFF FOR CODEX (machine-readable)

```yaml
top_priority_tasks:
  - goal: "Decouple index GO/NO-GO from unrelated macro feed staleness; make index-only runs evaluate only index-relevant gates."
    files_likely_touched:
      - "configs/quality_gates.index.yaml"
      - "configs/freshness.index.yaml"
      - "src/kalshi_alpha/core/gates/quality_gates.py"
      - "src/kalshi_alpha/exec/preflight_index.py"
      - "src/kalshi_alpha/exec/runners/scan_ladders.py"
      - "reports/_artifacts/go_no_go.json (artifact output)"
    acceptance_criteria:
      - "Running the index pipeline with stale macro feeds still yields GO when Polygon WS freshness + calibration freshness are within thresholds."
      - "GO/NO-GO reasons list is series-scoped and does not include macro namespaces when running index-only."
      - "Scoreboard reflects the new gating logic (no false NO-GO)."
    recommended_tests_or_commands:
      - "pytest -q"
      - "python -m kalshi_alpha.exec.preflight_index --now '2025-12-20T14:00:00-05:00'"
      - "python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run"

  - goal: "Implement a settlement-basis audit: compare Polygon index value at <time> vs Kalshi-documented expiration value for the same windows."
    files_likely_touched:
      - "tools/settlement_basis_audit.py (new)"
      - "src/kalshi_alpha/markets/discovery.py"
      - "src/kalshi_alpha/core/kalshi_api.py (or client wrapper used by discovery)"
      - "src/kalshi_alpha/exec/scanners/fast_index.py (optional reuse for snapshots)"
      - "reports/settlement_basis/*.md (new artifacts)"
    acceptance_criteria:
      - "Produces a daily report with basis distribution (mean/median/tails) per series (INXU/NASDAQ100U/INX/NASDAQ100)."
      - "Flags windows where basis could flip a nearby strike outcome (within one tick of strike boundaries)."
      - "Audit is reproducible from saved raw inputs (no hidden live-only dependencies)."
    recommended_tests_or_commands:
      - "python tools/settlement_basis_audit.py --day 2025-11-10 --series INXU"
      - "python tools/settlement_basis_audit.py --day 2025-11-10 --series NASDAQ100U"

  - goal: "Start collecting ladder top-of-book snapshots + execution telemetry suitable for fill probability calibration."
    files_likely_touched:
      - "src/kalshi_alpha/exec/telemetry/sink.py"
      - "src/kalshi_alpha/exec/supervisor_index.py"
      - "src/kalshi_alpha/brokers/kalshi/ws_client.py (if used)"
      - "src/kalshi_alpha/replay/fill_model.py"
      - "data/raw/kalshi/tob/*.jsonl (new data)"
    acceptance_criteria:
      - "For each index window, TOB snapshots are written at a consistent cadence with timestamps and ladder identifiers."
      - "Snapshots are sanitized and bounded in size (depth-limited) and survive restarts."
      - "replay/fill_model.py can ingest snapshots and output a fillprob payload for core/execution/fillprob.py."
    recommended_tests_or_commands:
      - "python -m kalshi_alpha.exec.supervisor_index --series INXU --record-tob"
      - "python -m kalshi_alpha.replay.fill_model --snapshots-dir data/raw/kalshi/tob --out data/reference/fillprob/INXU.json"

  - goal: "Enforce pilot safety at the broker boundary: maker-only, size caps, acknowledgement, live-broker requirement, kill-switch checks."
    files_likely_touched:
      - "configs/pilot.yaml"
      - "src/kalshi_alpha/exec/runners/scan_ladders.py"
      - "src/kalshi_alpha/brokers/kalshi/live.py"
      - "src/kalshi_alpha/exec/heartbeat.py"
      - "src/kalshi_alpha/core/execution/order_queue.py"
    acceptance_criteria:
      - "Attempting to run live without explicit acknowledgement fails closed."
      - "Maker-only enforcement rejects any order that would cross the spread."
      - "Kill switch engaged prevents all submits and replacements."
      - "Max contracts/order and max unique bins are enforced even under retries."
    recommended_tests_or_commands:
      - "pytest -q"
      - "python -m kalshi_alpha.exec.runners.scan_ladders --mode live --require-ack 'I_UNDERSTAND' (should fail without proper ack)"
      - "touch data/proc/state/kill_switch && run live mode (must not send orders)"

  - goal: "Add robust order reconciliation + cancel-all-on-startup/shutdown for index supervisor."
    files_likely_touched:
      - "src/kalshi_alpha/exec/state/orders.py"
      - "src/kalshi_alpha/brokers/kalshi/live.py"
      - "src/kalshi_alpha/exec/supervisor_index.py"
      - "src/kalshi_alpha/core/execution/order_queue.py"
      - "src/kalshi_alpha/exec/monitors/runtime.py"
    acceptance_criteria:
      - "On restart, supervisor fetches open orders, reconciles state, and cancels any order outside the active window."
      - "Outstanding orders state converges to exchange truth within one loop iteration."
      - "Monitor artifacts include explicit reconciliation status and open-order counts."
    recommended_tests_or_commands:
      - "pytest -q"
      - "python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run --simulate-restart"

  - goal: "Generate fresh, informative scoreboards and readiness reports that surface data availability (including 'fills=0') and model/ops health."
    files_likely_touched:
      - "src/kalshi_alpha/exec/scoreboard.py"
      - "src/kalshi_alpha/exec/reports/ramp.py"
      - "src/kalshi_alpha/exec/slo.py"
      - "reports/scoreboard_7d.md"
      - "reports/pilot_readiness.md"
    acceptance_criteria:
      - "Scoreboards include: fills count, maker/taker breakdown, expected vs realized EV lines, and freshness metrics for index feeds."
      - "Pilot readiness explicitly blocks progression unless minimum fills/sample size thresholds are met (configurable)."
      - "SLO metrics export includes heartbeat freshness and time-at-risk per series."
    recommended_tests_or_commands:
      - "python -m kalshi_alpha.exec.scoreboard --window-days 7"
      - "python -m kalshi_alpha.exec.reports.ramp --window-days 14"
      - "python -m kalshi_alpha.exec.slo --publish-cloudwatch"

  - goal: "Calibrate and persist fill-alpha + slippage curves only after real fills exist; wire them into proposal EV penalties."
    files_likely_touched:
      - "src/kalshi_alpha/core/execution/fillratio.py"
      - "src/kalshi_alpha/core/execution/slippage.py"
      - "src/kalshi_alpha/core/execution/defaults.py"
      - "configs/execution_defaults.yaml (if present) or reference JSON under data/reference/"
      - "src/kalshi_alpha/exec/runners/scan_ladders.py"
    acceptance_criteria:
      - "fillratio.tune_alpha() produces a persisted alpha with sample_size metadata."
      - "slippage.fit_slippage() produces a monotone curve persisted per series."
      - "scan_ladders proposals reflect fill/slippage penalties and show them in reports."
    recommended_tests_or_commands:
      - "python -m kalshi_alpha.core.execution.fillratio --tune-alpha --series INXU"
      - "python -m kalshi_alpha.core.execution.slippage --fit --series INXU"
      - "python -m kalshi_alpha.exec.runners.scan_ladders --series INXU --dry-run --show-penalties"

  - goal: "AWS readiness: package supervisor + monitors with stable process supervision, log rotation, and alerting on stale heartbeat/WS disconnects."
    files_likely_touched:
      - "docker/aws-jobs/Dockerfile"
      - "configs/systemd/*"
      - "configs/logrotate/*"
      - "src/kalshi_alpha/exec/monitors/cli.py"
      - "src/kalshi_alpha/exec/heartbeat.py"
      - "docs/runbooks/aws_ops.md (new)"
    acceptance_criteria:
      - "Supervisor runs under a supervisor (systemd/ECS) with automatic restart and bounded logs."
      - "Alerts fire on: heartbeat_stale, ws_disconnect_rate, and kill_switch_engaged."
      - "Runbook documents recovery steps and safe shutdown/cancel-all procedure."
    recommended_tests_or_commands:
      - "docker build -f docker/aws-jobs/Dockerfile ."
      - "python -m kalshi_alpha.exec.monitors.cli --runtime"
      - "logrotate -d configs/logrotate/kalshi_alpha"
```

[1]: https://www.cftc.gov/sites/default/files/filings/orgrules/24/11/rules1113248701.pdf "NASDAQ100 AMEND - Google Docs"
[2]: https://kalshi.com/docs/kalshi-fee-schedule.pdf "Fee Schedule for Oct 2025  - 10.1.25 Update"
[3]: https://help.kalshi.com/markets/market-maker-program?utm_source=chatgpt.com "Market Maker Program"
