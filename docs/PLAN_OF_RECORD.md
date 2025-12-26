# PLAN OF RECORD — kalshi-sys (Index Ladders: INX/INXU + NASDAQ100/NASDAQ100U)

Last updated: 2025-12-23  
Current gate status: **PAPER (dry-run only)**  
Primary objective: **prove realism + operational readiness** before risking capital.

This is a high-risk financial system. **Profitability is not assumed** and is not a deliverable for this phase.

---

## 0) Hard scope, constraints, and stop-the-line rules

### 0.1 Hard scope (non-negotiable)
We trade **ONLY** Kalshi index ladder markets for:
- **Hourly intraday**: `INXU`, `NASDAQ100U`
- **Daily close**: `INX`, `NASDAQ100`

Supported ladder types in scope:
- **Above/Below** (hourly intraday ladders)
- **Range** and/or **Above/Below** (daily close ladders)

Data source constraints:
- **Polygon.io only** (Indices Advanced + Stocks Advanced) via API + websocket.

No scope drift:
- No macro markets.
- No other underlyings.
- No additional data vendors.

### 0.2 Non-goals (explicitly out of scope for this plan)
- “Alpha discovery” without execution realism.
- Any “guaranteed profit” framing.
- Scaling size before fill + basis + ops evidence exists.

### 0.3 Stop-the-line conditions
Immediately halt/NO-GO if any of the following is true:
- Kill switch engaged (`data/proc/state/kill_switch`).
- Clock skew beyond threshold (must be surfaced in preflight + monitors).
- Calibration missing/stale beyond allowed age.
- Basis audit missing/stale or flip-risk flagged for the series/day being traded
  (reason codes: `basis_audit_missing`, `basis_audit_stale`, `basis_flip_risk`).
- A “maker-only” system attempts to cross the spread or behave like a taker.
- Supervisor cannot persist artifacts durably (disk full / no retention / no run manifests).

---

## 1) System framing: what we trade, how we think we win, what “winning” requires

### 1.1 Instruments (Kalshi index ladders)
- We trade **Kalshi contracts priced in $0.01–$0.99** (YES/NO style), sized as **$1 notional per contract**.
- We treat “price” as probability-like, but do **not** assume market is efficient.

Trading posture by gate:
- **PAPER**: read-only market data + simulated fills (or “no-fill”) is fine.
- **PILOT**: **maker-only, post-only**, tiny size, strict caps, fill measurement.
- **LIVE**: only after evidence; still cap and retain “fail-closed” posture.

### 1.2 Settlement mapping & basis risk (critical)
- **Settlement truth is Kalshi’s expiration value for the contract window**, not Polygon prints.
- Polygon provides **signal** and **real-time context**, but **must not** be treated as settlement truth.
- Therefore we require a **daily “settlement basis audit” artifact** per series/window, and we gate on it.

Definition:
- **Basis** = (Polygon reference series value at a comparable timestamp) − (Kalshi expiration/settlement reference value)
- **Flip risk** = probability that basis noise could flip ladder outcomes near strikes at your intended quoting distance.
Fail-closed rule:
- Daily basis audit JSON must match the **ET as-of date** for the run and include a valid `generated_at`
  timestamp on/after the start of that ET date. Missing/invalid JSON or a flip-risk flag blocks GO.

### 1.3 Decision cadence
- Decisions are **window-based** (US/Eastern-aware):
  - hourly windows for `INXU` / `NASDAQ100U`
  - daily close window for `INX` / `NASDAQ100`
- The 24/7 supervisor should:
  - pick the active/next window,
  - scan → propose → (dry/pilot/live) execute,
  - monitor + report,
  - persist artifacts,
  - sleep until next window.

### 1.4 Edge hypotheses (to be tested, not assumed)
We only have “edge” if ALL of these are true:
- **Probability edge**: our modeled distribution for index moves is better calibrated than market-implied odds.
- **Microstructure edge**: we can capture maker fills without paying taker costs (requires real fill probability + queue realism).
- **Operational edge**: we run reliably in the correct windows and don’t miss opportunities or quote the wrong contracts.

### 1.5 What “winning” requires (reality constraints)
Any credible EV claim must be robust to:
- **Fee model correctness** (including rounding + INX/NASDAQ100 special fees).
- **Fill realism** (maker fill curves must be measured, not guessed).
- **Cancel/replace behavior** (replacement throttles, queue position, and adverse selection).
- **Latency bounds** (clock skew, websocket staleness, ET alignment).
- **Calibration stability** (sigma_tod / drift with monitored ages).
- **Risk caps** that actually constrain exposure (defense-in-depth, broker fail-closed).

---

## 2) Minimal credibility checklist (what we must build/prove before PILOT)

Each item below must have: (a) a deterministic test, and (b) an artifact in `reports/` or `data/proc/` that can be reviewed.

### 2.1 Data ingestion correctness + freshness
Goal: “We know what time it is, and the index value is fresh.”

Must-have evidence:
- Polygon websocket staleness monitor (age of last tick/agg).
- Time alignment and clock skew guardrails.
- Index-only GO/NO-GO is scope-isolated from macro freshness; index freshness gating is controlled by `configs/freshness.index.yaml` (monitor inputs) and `configs/quality_gates.index.yaml` (gate thresholds).
- Polygon WS fallback respects `/v1/marketstatus/now` so closed/extended hours are logged as inactive (avoids false stale alarms).
- Freshness monitor consults `/v1/marketstatus/now` so closed/extended hours do not flag `polygon_index.websocket` as stale.

Commands (expected to exist):
- `python -m kalshi_alpha.exec.preflight_index`
- `make monitors`
- `make collect-polygon-ws` (or equivalent)
- `python -m kalshi_alpha.exec.market_status` (ops check for market + serverTime)

Required artifacts (minimum):
- `data/proc/health/polygon_ws_status.json` (last_update_ts, age_seconds, series)
- `reports/health/polygon_ws_<DATE>.md` (human-readable summary)
- Run manifest under `docs/agent_runs/<RUN_NAME>/`

### 2.2 Calibration & validation (sigma_tod, drift, PIT/CRPS)
Goal: “Our probability engine isn’t stale and isn’t obviously wrong.”

Commands (expected):
- `make calibrate-index`
- `make pilot-readiness`

Required artifacts:
- `data/proc/calibration/index/<SERIES>/<ASOF_DATE>.json` (or parquet)
- `reports/calibration/index_summary_<ASOF_DATE>.md`
- `reports/pilot_readiness_<ASOF_DATE>.md` includes calibration age flags

### 2.3 Fees: correctness + drift watch
Goal: “Fee math matches Kalshi schedule and cannot silently drift.”

Commands (expected):
- `pytest -q`
- `make fee-rules-watch`

Required artifacts:
- `reports/fees/fee_rules_hash_<DATE>.txt` (hash + config version)
- Unit test output proving rounding + edge cases

### 2.4 Settlement basis audit (Polygon vs Kalshi expiration value)
Goal: “Basis noise is measured and controlled.”

Commands (expected):
- `python tools/settlement_basis_audit.py --series INXU --day <YYYY-MM-DD>` (repeat per series)

Required artifacts:
- `reports/basis/<SERIES>/<YYYY-MM-DD>.md`:
  - per-window basis quantiles (p01/p05/p50/p95/p99)
  - strike flip risk summary for typical strike spacing and quote distances
  - “PASS/FAIL” vs configured thresholds
- `data/proc/basis/<SERIES>/<YYYY-MM-DD>.json` (machine-readable):
  - `series`, `asof_date`, `generated_at`, `sample_count`
  - `basis_quantiles` (p01/p05/p50/p95/p99)
  - `per_window_deltas` (window_id, n, mean, p05, p50, p95)
  - `flip_risk` (flag, rationale, thresholds)

Preflight staleness rule (fail-closed):
- `asof_date` must equal the ET date for the run.
- `generated_at` must be a valid ISO timestamp on/after 00:00 ET for that date.
- Missing/invalid JSON or `flip_risk.flag == true` blocks GO with reason codes above.

### 2.5 Maker-fill realism (TOB + quote intent → empirical fill curves)
Goal: “We can estimate maker fill probability vs price/time-to-expiry/queue.”

Commands (expected):
- `python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run --record-tob` (writes quote intents + TOB snapshots)
- `python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run --record-tob --telemetry-only` (NO-GO override for telemetry-only runs)
- `python tools/build_fillcalib_dataset.py --series INXU --from <DATE> --to <DATE>`
- `make pilot-readiness`

Required artifacts:
- `data/proc/telemetry/tob/<RUN_ID>.jsonl.gz`
- `data/proc/telemetry/quote_intents/<RUN_ID>.jsonl.gz`
- `data/proc/telemetry/runs/<RUN_ID>.json`
- `data/proc/fillcalib/curves_<ASOF_DATE>.json`
- `reports/fillcalib/<ASOF_DATE>.md` (sample counts + conservative curve)
- `reports/ops/telemetry_volume_<YYYY-MM-DD>.md`

Retention + bounds:
- Per-window cap: 256KB per stream (TOB + quote intents).
- Per-record caps: TOB 10KB, quote intents 2KB.
- Retention: 30 days (pruned via `python -m kalshi_alpha.exec.housekeep --keep-days 30`).

### 2.6 Execution safety & risk caps (fail-closed)
Goal: “Even if strategy logic is wrong, the broker boundary stops bad orders.”

Commands (expected):
- `pytest -q`
- `make live-smoke  # read-only`
- `python -m kalshi_alpha.exec.live_smoke`

Required artifacts:
- `reports/safety/live_smoke_<DATE>.md` (what checks ran + pass/fail)
- Evidence of:
  - maker-only rejects crossing orders
  - kill switch blocks submits
  - pilot caps enforced at broker layer

### 2.7 24/7 ops readiness (AWS)
Goal: “Supervisor runs unattended and is observable.”

Commands (expected):
- `python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run` (24h+)
- `make monitors`

Required artifacts:
- `reports/ops/supervisor_dryrun_<DATE>.md` (start/stop times, restarts, incident notes)
- Heartbeat artifacts in `data/proc/state/`
- CloudWatch log group receiving supervisor logs (AWS evidence recorded in run logs)

---

## 3) Gated progression: PAPER → PILOT → LIVE

### Gate A — PAPER (offline + live quotes, simulated fills)
Definition:
- Read live market + Polygon data
- Compute proposals
- Record TOB + quote intents
- **No live order placement** (dry broker only)

Entry criteria:
- Can run preflight and supervisor in dry-run with artifacts.

Exit criteria (must all be true for at least **5 consecutive market days**):
- Index-only GO/NO-GO is not blocked by macro freshness or non-index feeds.
- Basis audit runs daily and does not show unacceptable flip risk vs configured thresholds.
- Calibration age summary is produced and not stale.
- Telemetry is bounded (disk usage stable; retention working).
- No safety violations: no out-of-window actions, no “would-cross” maker attempts, no clock-skew violations.
- Scoreboard/pilot-readiness reports show meaningful data coverage (even if fills are still 0 in PAPER).

Canonical commands:
- `pytest -q`
- `python -m kalshi_alpha.exec.preflight_index`
- `python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run`
- `make monitors`
- `make pilot-readiness`

Required artifacts (per day):
- `reports/paper/<YYYY-MM-DD>/scoreboard.md`
- `reports/paper/<YYYY-MM-DD>/go_no_go.json`
- `reports/paper/<YYYY-MM-DD>/basis_summary.md`
- `reports/paper/<YYYY-MM-DD>/calibration_summary.md`
- `docs/agent_runs/<RUN_NAME>/RUN.md` (see docs/DOCS_AND_LOGGING_SYSTEM.md)

### Gate B — PILOT (tiny live, maker-only)
Definition:
- Live broker enabled with explicit acknowledgement
- **Maker-only (post-only)** orders only
- Tiny size: start with **1-lot**, strict daily loss/position caps, strict bin count caps
- Goal is **measurement**: fills, fee reconciliation, basis behavior

Entry criteria (must all be true):
- Gate A exit criteria satisfied.
- Broker safety tests prove maker-only and kill-switch enforcement.
- Basis audit thresholds configured and passing for the exact series/windows intended.
- Fill telemetry pipeline is working end-to-end (even if curves are conservative).

Exit criteria (to consider LIVE; not a promise of profit):
- Sufficient number of quote attempts and non-zero fills to estimate a fill curve with confidence bands.
- Fee reconciliation: observed maker/taker fees match modeled fee math and config.
- Ops stability: 1–2 weeks of unattended supervisor operation with clean restarts and reconciliation proof.

Canonical commands:
- `python -m kalshi_alpha.exec.preflight_index`
- `python -m kalshi_alpha.exec.supervisor_index --series INXU` (pilot config + live broker; exact flags/config depend on implementation)
- `make pilot-readiness`

Required artifacts:
- `reports/pilot/<YYYY-MM-DD>/ledger.csv` (or parquet) with fills + fees
- `reports/pilot/<YYYY-MM-DD>/expected_vs_realized.md`
- Updated `data/proc/fillcalib/curves_<ASOF_DATE>.json`

### Gate C — LIVE (scaled cautiously)
Definition:
- Increased size only after pilot evidence
- Strict monitoring, alerting, kill switch, and post-trade evaluation are mandatory
- Continuous “EV honesty” checks and replay parity are enforced

Entry criteria:
- Gate B exit criteria satisfied.
- Replay parity + EV honesty are enforced in CI and in runtime monitors.
- AWS alerting is wired and tested (on-call surface exists).

---

## 4) Roadmap horizons

### Next 1–2 weeks (validity + instrumentation)
Priority is correctness and realism, not model cleverness.

1) **Decouple index-only GO/NO-GO from macro freshness**
- Commands: `pytest -q`, `make monitors`, `python -m kalshi_alpha.exec.preflight_index`, `python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run`
- Artifact: go/no-go includes explicit `scope=index` and only scoped blockers.

2) **Promote settlement basis audit to a first-class gate**
- Commands: `python tools/settlement_basis_audit.py ...`, `python -m kalshi_alpha.exec.preflight_index`
- Artifact: daily `reports/basis/...` with flip-risk summary; preflight can fail closed when missing/stale.

3) **Start bounded TOB + quote-intent telemetry capture (PAPER-safe)**
- Commands: `make collect-polygon-ws`, `python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run`
- Artifact: bounded telemetry files + retention documented and proven.

4) **Make calibration age visibility unavoidable**
- Commands: `make calibrate-index`, `make pilot-readiness`
- Artifact: single summary file rendered in scoreboard + readiness.

5) **Hard-enforce index-only scope at entrypoints**
- Commands: `pytest -q`, `make paper_live_offline`
- Artifact: non-index families require explicit override.

6) **Clock skew guardrails + AWS NTP remediation**
- Commands: `pytest -q`, `make monitors`
- Artifact: skew monitor + clear NO-GO reason + runbook steps.

### Next 4–8 weeks (measured execution + controlled pilot)
1) Run tiny maker-only pilot to measure fills.
2) Calibrate fill/slippage models from pilot data.
3) Enforce replay parity + EV honesty in CI.
4) Operationalize AWS: dashboards, alerts, crash recovery drills.
5) Daily expected-vs-realized report loop.

### Longer-term (only after credibility)
- Model upgrades only if they improve calibration under strict leakage controls.
- Scaling to more windows/series only after ops + realism remain stable.

---

## 5) External dependency facts (changeable; record sources + retrieval date)

All items below were verified on **2025-12-23** and must be re-checked at least monthly (or when behavior changes).

### Kalshi fees (source: fee schedule PDF)
- General taker fee formula (rounded up to nearest $0.01):
  - `fee = ceil_to_cent(0.07 * C * P * (1 - P))`
- Maker fees may apply only to certain markets (rounded up):
  - `fee = ceil_to_cent(0.0175 * C * P * (1 - P))`
- Specific INX / NASDAQ100 fee factor (rounded up):
  - `fee = ceil_to_cent(0.035 * C * P * (1 - P))`
Source URL: https://kalshi.com/docs/kalshi_fee_schedule.pdf

**Implementation note:** confirm whether the 0.035 factor applies to taker fees only (likely) vs all trades, and whether maker fees apply to these markets. Do not guess—verify against realized `maker_fees` / `taker_fees` returned by the API.

### Kalshi tick size / price format
- Current minimum tick size is 1 cent; subpenny pricing may be introduced.
Source URL: https://docs.kalshi.com/trading/subpenny-pricing

### Kalshi API order placement fields we depend on
- Limit order creation supports `post_only` (maker-only), `time_in_force`, `expiration_ts`, `count`, `side` (yes/no), and price bounds (1–99).
Source URL: https://docs.kalshi.com/api-reference/trade-api/create-order

### Kalshi trading hours maintenance window
- Markets trade 24/7 except weekly maintenance window (Thursday 3–5 AM ET).
Source URL: https://help.kalshi.com/trading/fees-and-expiration/what-are-the-trading-hours

### Polygon websocket endpoints / latency tiers (Indices Advanced implications)
- Indices “Value” websocket topic may be delayed on Indices Advanced (doc shows 15-min delay).
Source URL: https://polygon.io/docs/websocket/indices/value
- Indices “Aggregates per Second” websocket topic is shown as real-time on Indices Advanced.
Source URL: https://polygon.io/docs/websocket/indices/aggregate-per-second

**Implementation note:** for trading decisions you must use a real-time feed; ensure the pipeline uses endpoints with real-time access under your plan.

### Polygon request limits / guidance
- Free tier has strict request limits; paying tiers are “unlimited” but throttling may occur; guidance suggests staying under ~100 req/sec.
Source URL: https://polygon.io/knowledge-base/article/what-are-the-api-request-limits

### Polygon/Massive websocket connection limits
- One simultaneous websocket connection per cluster; additional connections are billed separately.
Source URL: https://support.massive.com/en/articles/9729443-how-many-massive-websocket-connections-can-i-use-at-one-time

---

## 6) Canonical “run it” commands (must remain PAPER-safe unless explicitly piloting)
- Unit tests: `pytest -q`
- Preflight: `python -m kalshi_alpha.exec.preflight_index`
- Index supervisor dry-run: `python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run`
- Monitors suite: `make monitors`
- Pilot readiness report: `make pilot-readiness`
- Fee drift watch: `make fee-rules-watch`
- Read-only broker smoke: `make live-smoke` and `python -m kalshi_alpha.exec.live_smoke`
