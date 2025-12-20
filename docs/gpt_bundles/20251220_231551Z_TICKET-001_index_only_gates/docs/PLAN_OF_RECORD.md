# PLAN OF RECORD — Kalshi Index Ladder Trading (SPX/NDX)

Last updated: 2025-12-20  
Scope (HARD): INX / INXU / NASDAQ100 / NASDAQ100U ladders ONLY (hourly + daily close).  
Data source (HARD): Polygon (Indices Advanced + Stocks Advanced) via REST + WebSocket.  
Execution venue (HARD): Kalshi Exchange (limit orders; maker-first; strict safety gating).  
Profitability (HARD): NOT assumed. We prove realism first (fees, fills, slippage, basis, latency, ops).

---

## 0) Where we are today (ground truth summary)

Current repo status (from Prompt-1 diagnosis + `project_state` docs):
- Pipeline exists to scan/index-price/propose; execution surface exists (dry + live adapters).
- Evidence is currently NO-GO: no ledger fills in committed artifacts; readiness reports indicate insufficient_data / stale gating artifacts.
- Top blockers:
  - GO/NO-GO scope coupling (index runs blocked by unrelated macro feed staleness).
  - Settlement basis uncertainty (Polygon print != Kalshi settlement truth by default).
  - Fill model calibration gap (no measured maker fill curves; backtests risk optimism).
  - Ops wiring gap (24/7 supervisor + alerting not proven in artifacts).

---

## 1) System framing (what we trade + what “winning” requires)

### 1.1 Instruments
We trade Kalshi “index ladder” contracts for:
- Hourly intraday (“U” series): INXU, NASDAQ100U
- Daily close: INX, NASDAQ100

We treat each “window” independently:
- A target expiration timestamp (ET-aligned)
- A ladder: above/below (binary around a strike) and/or ranges (bins)

### 1.2 Settlement / basis reality (non-negotiable)
- Contract settlement is determined by Kalshi’s defined expiration value (Source Agency: Kalshi in contract terms).
- Polygon is an input feed, not settlement truth.
- Therefore: we MUST measure Polygon-vs-Kalshi basis at the exact window times we trade (basis audit becomes a gate).

### 1.3 Edge hypothesis (what could create EV AFTER fees with realistic fills)
We are not claiming this exists; this is the thesis to be tested:
- Probabilistic edge: our PMF/CDF for index level at window time is better calibrated than market-implied odds.
- Microstructure edge: maker-first execution captures favorable prices without paying taker fees (requires real fill probability, queue realism, cancel/replace discipline).
- Operational edge: reliable, correct, and timely quoting when others are degraded, without blowing up.

### 1.4 “Winning” requirements checklist (all must be true)
Fees
- Correct fee schedule per market family, including rounding.
- Correct maker vs taker classification per fill (do not assume maker fees are 0).

Fills / execution realism
- Maker fill probability must be measured on our own quotes (TOB snapshots + queue/replace behavior).
- Slippage/adverse selection must be bounded and reflected in EV.

Latency + timing
- Decision timestamps must be correctly ET-aligned; DST-safe.
- Data freshness thresholds must be met, and reconnect behavior must keep us fresh.

Calibration stability
- sigma_tod / drift curves must be fresh and stable; drift alarms must exist.
- Model must expose “honesty” (calibration diagnostics + sample size).

Risk caps
- Pilot requires hard caps (1-lot, maker-only, low notional risk, kill switch, strict window guard).
- Broker-boundary enforcement: even if strategy code goes wrong, broker wrapper must fail-closed.

Ops
- 24/7 supervisor must self-heal (crash recovery), persist state, and alert on stale heartbeat, stale WS, or any live orders outside windows.

---

## 2) Minimal credible system (what we must build/verify before risking real money)

### 2.1 Data correctness + freshness (Polygon)
Required:
- Polygon Index WebSocket connected; timestamps converted UTC → ET correctly.
- REST fallback for backfill and sanity checks.
- Freshness monitors + reconnection with bounded downtime.

Proof artifacts:
- `reports/_artifacts/monitors/freshness_*.json`
- `data/proc/state/heartbeat.json`
- Daily “feed health” markdown summary.

### 2.2 Market discovery correctness (Kalshi ladders)
Required:
- Correct mapping: window timestamp ↔ Kalshi market tickers ↔ ladder strikes/bins.
- Regression fixtures for strike extraction across DST week and a normal week.

Proof artifacts:
- `reports/_artifacts/discovery_parity/*.json`
- Tests freezing known tickers/windows.

### 2.3 Calibration + model validation
Required:
- sigma_tod + drift curves built from Polygon minute data per series/horizon.
- Calibration freshness gate enforced.
- Honesty diagnostics (PIT / calibration curves) with sample sizes.

Commands (existing entry points; see `PIPELINE_FLOW.md`):
- `python -m jobs.calibrate_hourly --series INXU NASDAQ100U`
- `python -m jobs.calibrate_close --series INX NASDAQ100`

Proof artifacts:
- `data/proc/calib/index/<symbol>/<horizon>/params.json`
- `reports/calib/` plots
- Drift monitor outputs under `reports/_artifacts/monitors/`

### 2.4 Fee + fill realism (EV honesty)
Required:
- Fee model applied using official formulas + rounding.
- Fill model starts conservative and is calibrated ONLY from measured data:
  - Log TOB snapshots and our own order placements.
  - Build empirical fill curves by price level, time-to-expiry, and replacement cadence.

Proof artifacts:
- `data/raw/kalshi/tob/*.jsonl`
- `data/proc/fillcalib/*.parquet` (derived)
- `reports/fillcalib/*.md` (sample sizes + curves)
- Explicit “fills=0” surfaced when no fills exist.

### 2.5 Execution safety + broker boundary enforcement
Required:
- Pilot config is enforced at the last mile (broker submit path), not just in strategy selection.
- Maker-only enforcement rejects crossing orders (post_only or equivalent).
- Acknowledgement gate + live-broker requirement + kill switch are fail-closed.

Proof artifacts:
- Unit/integration tests for:
  - “live without ack fails”
  - “crossing order rejected in pilot”
  - “kill switch blocks submits”
- `reports/_artifacts/go_no_go.json` includes pilot safety checks.

### 2.6 Monitoring + ops readiness
Required:
- Structured logs with run_id, window_id, market_ticker, order_id.
- Heartbeat + SLO metrics.
- Alerting hooks (Slack/email) for:
  - stale WS
  - stale heartbeat
  - order outside window
  - excessive cancel/replace
  - reconciliation mismatch

Proof artifacts:
- `reports/scoreboard_7d.md`
- `reports/pilot_readiness.md`
- `reports/_artifacts/monitors/*.json`

---

## 3) Gated progression: PAPER → PILOT → LIVE (with acceptance criteria)

### Gate A — PAPER
Definition:
- Uses real market discovery + real quotes (Kalshi public), but simulated fills (dry broker).
- Produces realistic “EV-after-fees-after-fillpenalty” proposals.
- Does NOT place any live orders.

Entry criteria:
- Index-only GO/NO-GO is functioning (macro feed staleness does not block).
- Calibration freshness checks pass for at least one series.

Exit criteria (must all pass):
- Daily run produces non-stale artifacts:
  - `reports/_artifacts/go_no_go.json` (series-scoped)
  - `reports/index_ladders/<HHMM>/...` reports
  - `reports/scoreboard_7d.md` updated
- EV accounting includes fees; outputs show fee per trade and rounding.
- Fill model is conservative-by-default and explicitly marked “uncalibrated” until telemetry exists.

Commands:
- Preflight:
  - `python -m kalshi_alpha.exec.preflight_index`
- Window scans:
  - `python -m kalshi_alpha.exec.scanners.scan_index_hourly --report`
  - `python -m kalshi_alpha.exec.scanners.scan_index_close --report`
- Supervisor (dry):
  - `python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run`
- Reporting:
  - `python -m kalshi_alpha.exec.scoreboard`
  - `python -m kalshi_alpha.exec.reports.ramp`

Expected artifacts:
- `reports/_artifacts/go_no_go.json`
- `reports/_artifacts/monitors/*`
- `reports/index_ladders/<HHMM>/*`
- `reports/scoreboard_7d.md`
- `reports/pilot_readiness.md`

### Gate B — PILOT (tiny risk, maker-only)
Definition:
- Live broker enabled with strict caps:
  - 1-lot per order (or minimal)
  - maker-only (post_only)
  - hard max daily cost / exposure
  - strict window guard
- Objective: measure fills and operational stability, not profits.

Entry criteria:
- PAPER gate stable for ≥ 5 consecutive market days with no safety violations.
- Settlement basis audit exists and is acceptable for the chosen series (see Gate C).

Exit criteria:
- Demonstrated operational stability:
  - no orders outside windows
  - reconciliation stable after restarts
  - bounded cancel/replace rate
- Measured fill dataset:
  - enough maker attempts to estimate fill probability by price level with confidence bands
  - explicit sample sizes in reports
- Fee reconciliation:
  - observed fees in fills match config/monitor expectations (within rounding behavior)

Commands:
- Supervisor live pilot:
  - `python -m kalshi_alpha.exec.supervisor_index --series INXU --mode pilot`
- Telemetry capture:
  - (ticketed) `--record-tob` or equivalent flag writing to `data/raw/kalshi/tob/`
- Reporting:
  - `python -m kalshi_alpha.exec.scoreboard`
  - `python -m kalshi_alpha.exec.reports.ramp`

Expected artifacts:
- `data/proc/ledger/*.jsonl` (real fills recorded)
- `data/raw/kalshi/tob/*.jsonl`
- `reports/fillcalib/*.md`
- `reports/pilot_readiness.md` (pilot-specific)

### Gate C — LIVE (only after evidence)
Definition:
- Scale cautiously from pilot; still maker-first with strict risk.
- LIVE is not a milestone; it’s a controlled extension of pilot after evidence.

Entry criteria (minimum):
- Pilot stability for ≥ 2 weeks.
- Settlement basis risk quantified and not able to systematically flip outcomes near strikes at our quoting distances.
- Fill model calibrated and EV accounting reflects measured fills + observed slippage.

Exit criteria:
- Continuous compliance + post-trade evaluation loop.
- Clear stop conditions and drawdown caps.

---

## 4) Roadmap horizons (implementation-ready priorities)

### Next 1–2 weeks (credibility: correctness + realism + instrumentation)
1) Fix index-only GO/NO-GO scope
- Decouple index pipeline from unrelated macro gates.
- Refresh scoreboards and readiness artifacts.

2) Settlement basis audit (Polygon vs Kalshi expiration value)
- Daily report, per series, with “strike flip risk” flags.

3) TOB + own-quote telemetry collection
- Persist snapshots for ladders we quote; derive first conservative fill curves.

4) Broker-boundary pilot safety enforcement
- Maker-only enforcement, ack gate, kill switch, strict caps.

5) Report truthfulness upgrades
- Every report must surface: sample sizes, fills=0, “uncalibrated” markers.

### Next 4–8 weeks (pilot evidence + ops hardening)
- Run 1-series pilot (pick INXU OR NASDAQ100U first).
- Robust reconciliation (open orders on startup/shutdown; cancel stale).
- Calibrate fill and slippage models from real outcomes.
- Add regression tests for discovery + strike alignment incl. DST boundaries.
- AWS wiring for supervisor + alerts (runbook + watchdog).

### Longer-term (optional after credibility)
- Model upgrades (regime conditioning, realized vol, intraday dynamics).
- Scaling (more windows/series) ONLY after reliability and realism prove out.

---

## 5) External facts to verify & track (record retrieval date + source)
Retrieval date for the following references: 2025-12-20

Kalshi
- Fee schedule PDF (index fees and rounding; verify maker fee applicability per market):  
  https://kalshi.com/docs/kalshi-fee-schedule.pdf
- Order types + API fields (limit orders, post_only, time_in_force):  
  https://docs.kalshi.com/api-reference/orders/create-order  
  https://docs.kalshi.com/fix/order-entry
- Tick size / subpenny transition (parse price_dollars fields now):  
  https://docs.kalshi.com/getting_started/subpenny_pricing

Polygon / Massive (Polygon docs redirect)
- Indices WebSocket overview (UTC timestamps; convert to ET; feeds):  
  https://polygon.io/docs/websocket/indices/overview
- Indices WebSocket aggregates-per-second (I: prefix; params; fields):  
  https://polygon.io/docs/websocket/indices/aggregates-per-second
- REST request limit guidance (free tier + recommended upper bounds):  
  https://polygon.io/knowledge-base/article/what-is-the-request-limit-for-polygons-restful-apis
- WebSocket connection limits (per cluster):  
  https://massive.com/knowledge-base/article/how-many-massive-websocket-connections-can-i-use-at-one-time

Policy:
- Any time these change, update `configs/fees.json` and write a note in `docs/PROGRESS.md` with the retrieval date and the delta.

---

## 6) Repo entry points & key modules (anchors)
(These are the “touch points” that define the index ladder system.)

Pipelines / runners
- `src/kalshi_alpha/exec/scanners/scan_index_hourly.py`
- `src/kalshi_alpha/exec/scanners/scan_index_close.py`
- `src/kalshi_alpha/exec/supervisor_index.py`
- `src/kalshi_alpha/exec/preflight_index.py`
- `src/kalshi_alpha/exec/runners/scan_ladders.py`

Pricing / fees / execution realism
- `src/kalshi_alpha/models/pmf_index.py`
- `src/kalshi_alpha/core/pricing/align.py`
- `src/kalshi_alpha/core/fees/*`
- `src/kalshi_alpha/core/execution/fillprob.py`
- `src/kalshi_alpha/core/execution/slippage.py`
- `src/kalshi_alpha/exec/quote_microprice.py`

Risk / gates
- `src/kalshi_alpha/core/gates/quality_gates.py`
- `src/kalshi_alpha/core/risk/*`
- `configs/quality_gates.index.yaml`
- `configs/freshness.index.yaml`
- `configs/pilot.yaml`
- `configs/fees.json`

Reporting / monitoring
- `src/kalshi_alpha/exec/scoreboard.py`
- `src/kalshi_alpha/exec/reports/ramp.py`
- `monitor/drift_sigma_tod.py`
- `monitor/fee_rules_watch.py`

---

## 7) Definition of “done” for credibility (not profits)
We consider the system “credible enough to pilot” only when:
- GO/NO-GO gates are correct and series-scoped
- Settlement basis is measured and bounded
- Fill curves are measured from our own quotes
- EV accounting includes fees + fill realism penalties
- Pilot safety is enforced at broker boundary
- Ops can run for weeks without babysitting
