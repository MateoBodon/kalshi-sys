# AGENTS.md – Kalshi Alpha (SPX/NDX Index Ladders)

This file tells AI coding agents (Codex CLI, Copilot, Cursor, etc.) how to work safely and effectively in this repository.

The **current priority** is:  
**SPX/NDX index ladders only** – `INX`, `INXU`, `NASDAQ100`, `NASDAQ100U`.

---

## 1. Project Context

- This repo is an execution + research monorepo for **Kalshi ladder markets**.
- We are focusing on **S&P 500 and Nasdaq-100 ladders**:
  - Intraday (noon/hourly) ladders.
  - Daily close ladders.
- Live trading must be:
  - **Maker-first**, with explicit fee modelling.
  - **Small size** (1-lot) until multi-month profitability is proven.
  - Fully under **PAL / VaR / GO–NO-GO** controls.

When in doubt, **opt for safety and observability over cleverness**.

---

## 2. Environment & Setup

Before doing any work:

1. Make sure you’re in the repo root (folder containing `pyproject.toml` and `README.md`).
2. Use Python 3.11+ in a virtualenv:

   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -e ".[dev]"
   ```

3. Secrets:
   - Never commit secrets.
   - Expect live configuration via `.env.local` (git-ignored) with keys like:
     - `KALSHI_API_KEY_ID`
     - `KALSHI_PRIVATE_KEY_PEM_PATH`
     - `POLYGON_API_KEY`
   - In tests, **do not** print env var names or values.

If you need additional tooling (e.g., `uv`, `ruff`, `mypy`), inspect `pyproject.toml`, `Makefile`, and GitHub Actions workflows to discover the canonical commands before adding new ones.

---

## 3. Repository Map (What Matters Most)

Prioritize understanding these paths:

- Core logic:
  - `src/kalshi_alpha/core/` – pricing, fees, PAL, VaR, backtest helpers.
  - `src/kalshi_alpha/strategies/index/` – SPX/NDX intraday + close models and PMF logic.
- Execution:
  - `src/kalshi_alpha/exec/` – scanners, pipelines, runners, brokers.
  - `src/kalshi_alpha/exec/state/` – outstanding orders, kill-switch, heartbeat.
- Schedules & monitoring:
  - `src/kalshi_alpha/sched/` (or similar) – trading windows & ET time handling.
  - `monitor/` – fee/rule watcher, sigma drift monitor, WS health.
- Data & reports:
  - `data/raw/` – raw data (Polygon, WRDS, etc., git-ignored).
  - `data/proc/` – processed/calibration data, state, ledgers.
  - `reports/` – Markdown reports, backtests, scoreboards.
- Tests:
  - `tests/` – unit/integration tests and fixtures for drivers, scanners, and strategies.

Do **not** assume other strategy families (CPI, claims, weather, gas) are actively maintained; treat them as **secondary** unless explicitly told otherwise.

---

## 4. Build, Lint, and Test

**Family switch:** Pipelines and scoreboard now default to `FAMILY=index` (SPX/NDX only). When you intentionally need CPI/claims/weather/other macro runs, pass `--family macro` (or set `FAMILY=macro`) to `daily`/`today`/`week` pipelines and related scripts.

### 4.1 How to find the canonical commands

Before running anything heavy, **inspect**:

- `.github/workflows/*.yml`
- `pyproject.toml`
- `Makefile`

Use `rg` to search for test and lint commands, e.g.:

```bash
rg "pytest" .github Makefile pyproject.toml
rg "ruff" .github Makefile pyproject.toml
rg "mypy" .github Makefile pyproject.toml
```

Prefer the same commands CI uses.

### 4.2 Minimum test protocol for any non-trivial change

For any significant code change, run at least:

1. **Unit tests**:

   ```bash
   pytest
   ```

   or the more specific command used in CI (e.g., `pytest tests/strategies tests/exec`).

2. **Static checks** (if configured):

   ```bash
   ruff check .
   mypy src tests
   ```

3. **Index ladder offline smoke tests**:

   - Hourly:

     ```bash
     python -m kalshi_alpha.exec.scanners.scan_index_hourly --offline --fixtures
     ```

   - Close:

     ```bash
     python -m kalshi_alpha.exec.scanners.scan_index_close --offline --fixtures
     ```

   - Quick loop (CI-friendly) using trimmed Polygon fixtures:

     ```bash
     python -m kalshi_alpha.exec.scanners.scan_index_hourly --offline --fast-fixtures
     python -m kalshi_alpha.exec.scanners.scan_index_close --offline --fast-fixtures
     ```

4. **Time-awareness gate checks** (sched/window/final-minute changes):

   ```bash
   PYTHONPATH=src pytest tests/exec/test_time_awareness.py
   ```

5. **Scoreboard / reporting smoke**:

   ```bash
   python -m kalshi_alpha.exec.scoreboard --family index --offline
   ```

### Polygon-only index modelling/backtest helpers

- Build the Polygon minute panel (expects raw parquet under `data/raw/polygon/index/` or legacy `data/raw/polygon/I_SPX`):

  ```bash
  python scripts/build_index_panel_polygon.py --input-root data/raw/polygon/index --output data/proc/index_panel_polygon.parquet
  ```

- Fit the simple Polygon model params:

  ```bash
  python -m jobs.calibrate_index_polygon_model --panel data/proc/index_panel_polygon.parquet --series INX INXU NASDAQ100 NASDAQ100U --output-root data/proc/calib/index_polygon
  ```

- Run the offline backtest:

  ```bash
  PYTHONPATH=src python -m kalshi_alpha.exec.backtest_index_polygon --panel data/proc/index_panel_polygon.parquet --params-root data/proc/calib/index_polygon --series INXU NASDAQ100U --start-date YYYY-MM-DD --end-date YYYY-MM-DD
  ```

- Fast tests for the new stack:

  ```bash
  PYTHONPATH=src pytest tests/strategies/test_index_panel_polygon.py tests/strategies/test_model_polygon.py tests/exec/test_backtest_index_polygon.py
  ```

If these commands don’t exist or fail unexpectedly, **inspect the relevant modules and CI configs**, fix the issue, and update this AGENTS.md section accordingly.

---

## 5. Real Data vs Synthetic Data

We strongly prefer **real data** over synthetic for anything that touches trading logic.

### 5.1 WRDS and Polygon data

- Historical index data:
  - WRDS / other institutional feeds should be stored under `data/raw/index_wrds/` or similar.
  - Polygon minute bars and snapshots live under `data/raw/polygon_index/`.
- Processed panels:
  - Expect combined panels under `data/proc/index_panel.parquet` (or similar).
  - Calibration parameters under `data/proc/calib/index/...`.

### 5.2 Guidelines for agents

- **Calibration / backtesting / EV logic**:
  - Use WRDS + Polygon + Kalshi historical data, not synthetic data.
  - Synthetic data is acceptable only for:
    - Narrow unit tests that don’t depend on distribution shape.
    - Edge-case testing of error handling, not live EV predictions.
- When adding tests:
  - Prefer fixtures built from real historical data stored under `tests/data_fixtures/index/` or similar.
  - Avoid hardcoding unrealistic toy prices or vol levels.

If you don’t see WRDS data available, document the gap and use recorded Polygon/Kalshi data instead.

---

## 6. Trading & Risk Guardrails (for Execution Changes)

When working on anything that can affect **live orders**:

1. Treat **DryBroker** as the default.
2. Live trading requires:
   - `--broker live`
   - Explicit confirmation flags (e.g., `--i-understand-the-risks`).
   - Presence of `.env.local` with keys.
3. Never remove or bypass:
   - Kill-switch.
   - GO/NO-GO checks.
   - PAL / VaR / loss caps.
4. For index ladders specifically:
   - Default to **maker-only**, 1-lot, ≤2 bins per market.
   - Respect fee schedule for indices.
   - Never ship code that silently flips to taker without an explicit CLI flag.

If a change would weaken these protections, **stop** and either:
- Propose a safer design, or
- Ask for explicit human approval via an issue/comment.

---

## 7. Codex CLI–Specific Instructions

These rules are for Codex CLI or any similar coding agent.

### 7.1 Session startup

On starting a session in this repo:

1. Run:

   ```bash
   git status
   ```

   to understand the current worktree state.

2. Read:
   - `AGENTS.md` (this file).
   - `README.md`.
   - `REPORT.md` and `reports/pilot_readiness.md` if present.
   - `kalshi_alpha_long_term_plan.md` (if present) for strategic context.

3. Summarize your understanding in a scratch file:

   ```text
   report/agent_logs/YYYYMMDD_codex_notes.md
   ```

   - Briefly note what you plan to do in this session.

### 7.2 How to work

- Favor small, incremental changes with clear intent.
- Use `apply_patch` (or equivalent) for file edits when available.
- Prefer `rg` over `grep` to search code.
- Index websockets: use `kalshi_alpha.drivers.polygon_index_ws` (`polygon_index_ws` context +
  `close_shared_connection`) so each process maintains a single Massive index WS; metrics helpers
  `active_connection_count`/`last_message_age_seconds` are available for monitoring.
- When adding new modules or commands:
  - Wire them into tests and CI where appropriate.
  - Add short docstrings and comments only where the logic is non-obvious.

### 7.3 Tests and documentation before concluding

Before you consider a session “done”:

1. Run the **minimum test protocol** in Section 4.
2. If tests are long or expensive:
   - At least run scope-limited tests relevant to your changes.
3. Update documentation where necessary:
   - `AGENTS.md` if you change workflows for agents.
   - `README.md` or `docs/` if you change user-facing behavior.
4. Append a short summary of:
   - What you changed.
   - How you tested it.
   - Any open follow-ups.
   to `report/agent_logs/YYYYMMDD_codex_notes.md`.

### 7.4 Git, commits, and pushes

- Always keep commits focused and descriptive:
  - Example messages:
    - `index: tighten sigma_tod drift thresholds`
    - `exec: add preflight_index script`
- Never commit:
  - `.env`, `.env.local`, or any secrets.
  - Large raw data files unless explicitly allowed.
- If allowed by the harness and configuration:
  - You may run `git commit` and `git push` at the end of a successful session.
  - If you are unsure, **leave changes staged/committed locally** and describe what a human should run.

---

## 8. AWS & Remote Environments

If you are running **inside an AWS compute environment** (ECS, Batch, EC2, etc.):

- Treat the filesystem as ephemeral:
  - Don’t stash long-term artifacts outside `data/` or `reports/`.
- Don’t assume `.env.local` exists; instead:
  - Read secrets from env vars or injected secret managers.
- Prefer making **job entrypoints** idempotent:
  - Running the same job twice should not double-send orders.
- Log clearly to STDOUT / STDERR using structured JSON where practical.

---

## 9. What *Not* to Do

- Don’t:
  - Remove or neuter risk checks to “see what happens live”.
  - Add hidden flags that bypass safeguards.
  - Introduce synthetic test data that misrepresents real index behavior into core calibration or EV code.
  - Dump large file contents into the terminal or this file.

If you are uncertain about a change’s risk implications, **stop and document** your concerns in `report/agent_logs/` instead of guessing.

---

Follow these instructions and you will be a good citizen in this repo: safe, reproducible, and focused on building a robust SPX/NDX ladder trading engine.
