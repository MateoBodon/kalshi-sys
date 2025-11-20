# AGENTS.md - Kalshi Alpha Automated Trading System

> **WARNING FOR AGENTS:** This system handles real capital. Safety, latency, and correctness are paramount. Do not hallucinate APIs.

## 1. Project Overview
`kalshi-sys` is a Python 3.11 monorepo for algorithmic trading on Kalshi (prediction markets).
- **Core Assets:** SPX (S&P 500) and NDX (Nasdaq-100).
- **Markets:** Intraday Hourly (e.g., `KXINXU-25NOV03H1200`) and Daily Close (e.g., `KXINX-25NOV03H1600`).
- **Data Source:** Polygon.io (Indices Advanced + Stocks Advanced).
- **Execution:** `LiveBroker` (Real $) and `DryBroker` (Paper).

## 2. Command Protocol
* **Install:** `pip install -e ".[dev]"`
* **Test:** `pytest -q tests/` (Must pass before commit)
* **Lint:** `ruff check .`
* **Typecheck:** `mypy src/`
* **Run Scan (Dry):** `python -m kalshi_alpha.exec.runners.scan_ladders --series INXU --offline --report`
* **Start Supervisor:** `python -m kalshi_alpha.exec.supervisor --broker live --sniper --ack-risks` (adds 24/7 hourly/EOD scheduling + kill-switch guard)
* **Cloud Job:** `python scripts/aws_job.py --job calibrate_hourly`

## 3. Architecture Rules
1.  **Monorepo Structure:** All source code is in `src/kalshi_alpha`.
2.  **No Hardcoding:** Secrets must be loaded from `.env.local` or environment variables via `kalshi_alpha.utils.env`.
3.  **Safety Gates:**
    * **Kill Switch:** Check `data/proc/state/kill_switch` before *every* order.
    * **Freshness:** Data older than 2 seconds is "stale". Do not trade on stale data. Supervisor auto-trips the kill switch if Polygon WS latency/age >500 ms.
    * **Limits:** Respect `configs/pal_policy.yaml` (Position limits).

## 4. Development Workflow
1.  **Plan:** Before writing code, analyze the `reports/` to see how the strategy is currently performing.
2.  **Implement:** Write code in small, testable chunks.
3.  **Verify:** Use `tools.replay` to simulate your changes against yesterday's market data.
4.  **Document:** Update `CHANGELOG.md` with specific metrics (e.g., "Improved latency by 50ms").

## 5. Key Files
- `src/kalshi_alpha/strategies/index/hourly_above_below.py`: The math (Gaussian/Skew models).
- `src/kalshi_alpha/exec/runners/scan_ladders.py`: The main execution loop.
- `src/kalshi_alpha/exec/supervisor.py`: 24/7 daemon scheduling hourly scans, close scans, and websocket freshness guard.
- `src/kalshi_alpha/exec/heartbeat.py`: System health monitoring.
