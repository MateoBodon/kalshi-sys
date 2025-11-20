# Kalshi Alpha – Long-Term Repo Plan (SPX/NDX Index Ladders Only)

This document is a concrete roadmap for the `kalshi-sys` / “Kalshi Alpha” repo focused **only** on the four index ladder series:

- `INX` – S&P 500 close ladders  
- `INXU` – S&P 500 intraday (noon/hourly) ladders  
- `NASDAQ100` – Nasdaq-100 close ladders  
- `NASDAQ100U` – Nasdaq-100 intraday (noon/hourly) ladders  

The goal is to build a **safe, fully-automated, research‑driven system** that:

1. Continuously monitors these four series.
2. Uses **real data** (Polygon + WRDS + Kalshi) to calibrate probabilities.
3. Trades **maker‑first**, fully fee‑aware, with strict risk guardrails.
4. Runs primarily on **AWS** for production pipelines.
5. Is friendly to **Codex CLI**, with clear instructions in `AGENTS.md`.

Everything else (macro ladders, weather, gas, etc.) is **secondary** for now.

---

## 1. Scope, Constraints, and Principles

### 1.1 Trading scope

- **In-scope**
  - Only SPX/NDX ladder contracts:
    - `INX`, `INXU`, `NASDAQ100`, `NASDAQ100U`.
  - Intraday hourly windows during US cash hours (9:30–16:00 ET).
  - Daily close ladders (16:00 ET).
  - Maker‑only by default; taker fills only when explicitly allowed and EV after fees comfortably clears a strict threshold.

- **Out-of-scope for now**
  - CPI, claims, treasury, weather, gas strategies.
  - Non‑index Kalshi markets (elections, politics, sports, random stuff).
  - Fancy cross‑venue arbitrage; keep it single‑venue (Kalshi) until index ladder alpha is proven.

### 1.2 Design principles

1. **Risk first**
   - Never sacrifice risk controls for a bit more theoretical EV.
   - Maker‑only, 1‑lot, ≤2 bins per market until multi‑month profitability is demonstrated.

2. **Single source of truth**
   - One config that defines which series are active, what windows exist, and what risk caps apply.
   - No scattered hardcoded times, tickers, or “magic numbers”.

3. **Everything is reproducible**
   - Every production pipeline must be runnable offline using fixtures or archived data.
   - WRDS/Polygon/Kalshi data should be stored and versioned so you can re‑run historical experiments byte‑for‑byte.

4. **Codex‑friendly**
   - Codex should be able to:
     - Inspect the project quickly.
     - Discover how to run tests.
     - Run backtests and scans.
     - Make small, safe, well‑documented changes.
   - `AGENTS.md` carries explicit instructions for agents.

5. **AWS‑first for heavy work**
   - All production pipelines (calibrations, live scans, backtests, scoreboards) should be runnable on AWS.
   - Local machine is for exploration, notebooks, and small dev loops.

---

## 2. Phase 0 – Repo Hygiene & Index‑Only Focus

**Objective:** Clean up and hard‑focus the codebase so that **index ladders are the only first‑class citizens** for now.

### 2.1 Config & secrets

Tasks:

- Ensure `.env.local` exists (never committed) and is the **only** place Codex or humans expect sensitive values in dev:
  - `KALSHI_API_KEY_ID`
  - `KALSHI_PRIVATE_KEY_PEM_PATH`
  - `POLYGON_API_KEY`
  - Optional WRDS credentials (if needed by any drivers).
- `src/kalshi_alpha/utils/env.py` (or equivalent) should:
  - Load `.env.local` first, then `.env`, then actual env vars.
  - Never print raw env var names or values; errors must be generic.
- Add sanity tests:
  - `tests/dev/test_env_sanity.py`:
    - Fails if code ever prints known secret var names.
    - Confirms `.env.local` is git‑ignored and optional.

### 2.2 “Index family only” switch

Introduce an explicit **family filter**:

- New config file: `configs/family_index_only.yaml`:
  - Lists allowed series: `INX`, `INXU`, `NASDAQ100`, `NASDAQ100U`.
  - Declares that all other families are disabled for now.
- Pipeline changes:
  - `exec/pipelines/daily.py`, `exec/pipelines/today.py`, `exec/pipelines/week.py` should accept a `--family index` argument.
  - With `--family index`, pipelines:
    - Only run index calibrations.
    - Only run index scanners/runners.
    - Only update index scoreboards/reports.
- Tests:
  - Add `tests/pipelines/test_family_index_only.py` ensuring:
    - `--family index` skips CPI/claims/weather/gas.
    - No unexpected strategies execute.

### 2.3 CI profile for index‑only

Adjust GitHub Actions:

- Add a dedicated workflow `ci-index.yaml` (if not already present) that:
  - Sets `FAMILY=index` env var.
  - Runs:
    - `pytest` on index‑related tests.
    - Any `ruff` / `mypy` checks.
    - A small offline index pipeline smoke test (e.g., `python -m kalshi_alpha.exec.scanners.scan_index_hourly --offline --fixtures`).
- Ensure CI fails if:
  - Index calibrations cannot be loaded.
  - Index drivers (Polygon fixtures) fail to parse.

### 2.4 Documentation

- Update `README.md`:
  - Clearly state current focus on `INX`/`INXU`/`NASDAQ100`/`NASDAQ100U`.
  - Move all other strategies into a “Paused / not under active development” section.
- Add a short `docs/index_ladders/overview.md` summarizing:
  - What problems the index ladder stack solves.
  - High‑level pipeline: data → calibrations → scans → execution → scoreboard.

---

## 3. Phase 1 – Time & Automation (Know When to be Alive)

**Objective:** Make the system **self‑aware of time**, Kalshi trading hours, and index contract times so it knows **exactly** when to be active.

### 3.1 Canonical window definitions

- Implement or refine a central module: `kalshi_alpha.sched.windows` (already referenced in README).
- For each series, define:
  - `event_time_et`: e.g., `12:00` for `INXU`/`NASDAQ100U`, `16:00` for `INX`/`NASDAQ100`.
  - `scan_window_start_et` and `scan_window_end_et`:
    - For example: 11:50–11:59 ET for noon ladders, 15:50–15:59 ET for close ladders.
  - `final_minute_guard_start_et`: e.g., 11:59:00 and 15:59:00 ET.
- Use **US/Eastern** as the timezone everywhere; no naive datetimes in execution code.

### 3.2 Websocket & freshness monitoring

- Ensure index price data sources (Polygon WS, Kalshi WS if used) are monitored:
  - A small `monitor/ws_health_index.py` that:
    - Tracks last message timestamp per source.
    - Writes freshness metrics to heartbeat and SLO structures.
- Extend/validate `ws_final_minute_guard`:
  - Enforce a configurable max latency (e.g., 700 ms) in the final minute per series/window.
  - If violated:
    - Trigger kill‑switch at the runner level.
    - Log details (latency histogram, last price, last timestamp) for post‑mortems.

### 3.3 Time‑aware pipelines

- Update index runners:
  - `pilot_hourly.py` and `pilot_close.py`:
    - Must consult `sched.windows` before doing anything:
      - If outside the active window → exit early with an informative log.
    - Must support `--snap-to-window {wait,print}` semantics:
      - `wait`: block until near window start, then run.
      - `print`: log next planned window, then exit.
- Add tests:
  - `tests/exec/test_index_windows.py`:
    - Mocks ET timestamps and checks that:
      - Runners do not arm outside windows.
      - Runners arm only inside windows and obey `snap-to-window`.

### 3.4 AWS scheduling layout

- Define one AWS job per window/series, but share a single code path:
  - Example EventBridge / cron pattern:
    - Noon ladders: trigger at 11:50 ET, job uses `--snap-to-window wait`.
    - Close ladders: trigger at 15:50 ET, same.
- Document mapping in a new `docs/index_ladders/aws_windows.md` file:
  - Show each series, window, cron, and expected behavior.
  - Use this as reference for Codex when it needs to modify infra or job scripts.

---

## 4. Phase 2 – Data & Modelling (WRDS + Polygon + Kalshi)

**Objective:** Build a **robust distribution model** for SPX/NDX at intraday and close horizons that uses **real WRDS data** and Polygon, not synthetic toy data.

### 4.1 Data ingestion

- Create a new module `drivers/index_history_wrds.py` (or similar) that:
  - Pulls historical SPX/NDX levels (e.g., from WRDS CRSP/Compustat or similar).
  - Normalizes timestamps to ET.
  - Writes clean parquet under `data/raw/index_wrds/`.
- Add a fast aggregator:
  - `scripts/index_build_panel.py`:
    - Joins WRDS daily close data with Polygon intraday data for recent years.
    - Produces a combined panel `data/proc/index_panel.parquet` with:
      - Date
      - Time
      - Index level
      - Realized intraday variance so far
      - Flags for event days (CPI, FOMC, etc.), if available.

### 4.2 Feature engineering

- Add `strategies/index/features.py`:
  - Functions that take “now” and produce features for each target horizon:
    - `Δt` to target (minutes).
    - Realized intraday volatility up to now.
    - Overnight gap.
    - Day‑of‑week, month, etc.
    - Event‑day flags (if you have macro calendar data).
- Tests:
  - `tests/strategies/test_index_features.py` using WRDS fixtures and Polygon fixtures.

### 4.3 Distribution model

- Implement a **baseline heavy‑tail model** in `strategies/index/model.py`:
  - Use empirical CDFs or a mixture-of-normals / t‑distribution for ΔS over Δt.
  - Condition on features where possible.
  - Functions:
    - `fit(params, data)` – calibrate from historical panel.
    - `predict_pmf(now_state, params, ladder_strikes)` – output a PMF for ladder bins.
- Persist calibration:
  - Existing calibration jobs (`jobs.calibrate_hourly`, `jobs.calibrate_close`) should:
    - Read WRDS/Polygon panel.
    - Fit model parameters.
    - Save JSON parameters under:
      - `data/proc/calib/index/INX/{hourly,close}/params.json`
      - `data/proc/calib/index/NASDAQ100/{hourly,close}/params.json`

### 4.4 Calibration validation

- Add `tests/strategies/test_index_calibration.py`:
  - Ensures:
    - Calibration jobs run offline with fixtures.
    - PMF integrates to ~1.
    - Basic monotonicity and sanity checks around tails.
- Extend `monitor/drift_sigma_tod.py`:
  - Compare realized vol vs model’s implied vol per window.
  - Flag when drift exceeds a threshold and ensure execution layer shrinks size.

---

## 5. Phase 3 – Backtesting & EV Honesty

**Objective:** Build a **realistic backtest harness** that simulates index ladder trading with:

- Real historical data.
- An approximation of historical Kalshi orderbooks.
- Fee schedule and fill probabilities.

### 5.1 Ladder simulation module

- Create `strategies/index/backtest.py`:
  - Input:
    - Historical index paths.
    - Historical ladder strike grids for `INX`, `INXU`, `NASDAQ100`, `NASDAQ100U` (if available).
    - Model parameters (from calibration).
  - Output:
    - Simulated trades:
      - Ladder bin, price, side, size.
      - Theoretical EV, realized PnL, fees, slippage.
- If you don’t have full historical orderbooks:
  - At minimum, load historical Kalshi prices and approximate spreads.
  - Later, refine when you acquire richer data.

### 5.2 Fill & slippage model

- Extend `core/execution` with a basic fill model:
  - Given:
    - Relative quote vs mid.
    - Time left to expiry.
    - Depth profile (approximate).
  - Estimate:
    - Probability of being filled as maker.
    - Likely slippage for taker.
- Use this model both in backtests and in live EV calculations:
  - EV after fees **must** include:
    - Fee schedule.
    - Expected slippage.
    - Partial fill probabilities.

### 5.3 Backtest CLI

- Add a CLI tool: `kalshi-backtest-index` (thin wrapper around backtest module):
  - Options:
    - `--series {INX,INXU,NASDAQ100,NASDAQ100U,all}`
    - `--start-date`, `--end-date`
    - `--mode {maker-only,mixed}`
    - `--max-bins`, `--ev-threshold`
  - Produces:
    - CSV of trades.
    - Markdown summary under `reports/index_backtests/`.

### 5.4 Scoreboard integration

- Extend `exec/scoreboard.py`:
  - Include **backtest results** alongside live/paper results.
  - EV honesty metrics:
    - Compare model EV vs realized PnL in backtests.
  - This gives you pre‑live evidence before risking real money.

---

## 6. Phase 4 – AWS “Always On” Index Pipelines

**Objective:** Run all index workloads on AWS with **clean separation** between research, paper trading, and live execution.

### 6.1 Containerization

- Under `docker/aws-jobs`:
  - Ensure a Dockerfile that:
    - Uses Python 3.11.
    - Installs repo via `pip install -e ".[dev]"`.
    - Provides an entrypoint script, e.g., `/app/entrypoint.sh`.
- Entry point should support subcommands:
  - `index-calibrate-hourly`
  - `index-calibrate-close`
  - `index-scan-hourly`
  - `index-scan-close`
  - `index-backtest`
- Each command must:
  - Load `.env` from AWS Secrets Manager or task env vars (not from `.env.local`).
  - Respect `family=index`.

### 6.2 Job definitions

- For each of the four time‑of‑day windows (noon + close) and two series families (SPX/NDX):
  - Define an AWS Batch / ECS or similar task with:
    - A small reserved CPU/memory footprint.
    - Environment variables for `FAMILY=index`, `BROKER=dry|live`.
    - A strict timeout (e.g., 10–15 minutes per job).
- EventBridge schedule:
  - Noon jobs triggered at 11:50 ET.
  - Close jobs triggered at 15:50 ET.
  - Calibration jobs once per day (after market close).

### 6.3 Preflight & GO/NO‑GO gating

- Implement a `preflight_index.py` script:
  - Checks:
    - GO/NO‑GO state.
    - Heartbeat freshness.
    - Calibration age.
    - WRDS and Polygon connectivity (if online).
  - Returns non‑zero exit code if any gate fails.
- All AWS jobs:
  - Run `preflight_index.py` first.
  - Abort gracefully (no scans, no orders) if preflight fails.

### 6.4 Logging & metrics

- Ensure:
  - All jobs log to CloudWatch with structured JSON.
  - SLO metrics are published after each scan using `exec.slo`.
- Document log group names and metric namespaces in `docs/index_ladders/aws_monitoring.md`.

---

## 7. Phase 5 – Paper Trading & Live Pilot

**Objective:** Run a **disciplined live pilot**:

- First paper‑only.
- Then 1‑lot maker‑only live.
- Always with well‑defined metrics and risk limits.

### 7.1 Paper trading (BROKER=dry)

- Configure:
  - `BROKER=dry`
  - `FAMILY=index`
- Run:
  - All AWS jobs as usual.
  - Paper trades recorded in a ledger (`data/proc/ledger/index_paper.jsonl`).
- Scoreboards:
  - Weekly review of:
    - EV vs realized PnL (paper).
    - Fill probability predictions vs simulated fills.
    - Calibration drift.

### 7.2 Small live pilot

- Switch a subset of windows to `BROKER=live`:
  - Start with noon ladders only, 1‑lot maker‑only.
- Live risk caps:
  - Per‑day and per‑week max loss.
  - Per‑window max orders.
  - PAL policies on per‑strike loss.
- Every live run:
  - Must pass preflight.
  - Must log GO/NO‑GO status at start and end.

### 7.3 Evaluation criteria

- Do not scale up until:
  - Multi‑month live or semi‑live (live/paper mixed) PnL > 0.
  - EV honesty near 1.0 (within noise).
  - Maximum drawdown is within tolerance.
  - No “unknown unknowns” (e.g., rule misinterpretations, fee surprises).

---

## 8. Phase 6 – Scaling & Future Expansion

**Objective:** Once index ladder alpha is proven and stable, **carefully** expand.

### 8.1 Scaling within SPX/NDX

- Increase:
  - Number of bins per market (from 2 to 3–4 where justified).
  - Contracts per bin.
- Strengthen risk controls:
  - Additional PAL layers by moneyness.
  - Adaptive sizing based on realized PnL and volatility.

### 8.2 New families (later)

- Re‑enable macro ladders (CPI/claims/etc.) **only** when:
  - Index ladder engine is stable.
  - You have time to manage more complexity.
- Add additional equity indices or crypto ladders only if:
  - Data quality and liquidity are comparable.
  - You can reuse the same modelling stack with minimal changes.

---

## 9. Codex & Agent Workflow Integration

**Objective:** Make the repo easy for Codex CLI and other agents to extend.

### 9.1 AGENTS.md

- Maintain a clear `AGENTS.md` in repo root that:
  - Describes how to:
    - Set up environment.
    - Run tests.
    - Run index pipelines and backtests.
    - Avoid touching secrets.
  - Tells agents to:
    - Prefer real WRDS data in integration tests.
    - Avoid synthetic data for calibration/backtest work.
    - Keep commits small and well‑documented.

### 9.2 Agent workflow

Encourage the following pattern for Codex:

1. Read `AGENTS.md`, `README.md`, and this plan.
2. Summarize the current repo state in a scratch Markdown file under `report/agent_logs/`.
3. Implement **one phase or sub‑phase at a time**, ending with:
   - Passing tests.
   - A short Markdown summary of work in `report/agent_logs/YYYYMMDD_codex_run.md`.
   - A clean git commit with a descriptive message.

---

This plan gives you a **single source of truth** for how the repo should evolve over the next several months while staying focused on SPX/NDX ladders and keeping risk and reproducibility front and center.
