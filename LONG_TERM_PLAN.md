# kalshi-sys — Long-Term Plan (SPX/NDX Intraday & Daily-Close Ladders)

**Owner:** Mateo (kalshi-sys)  
**Scope:** SPX/NDX intraday (“U” markets, e.g., INXU/NASDAQ100U at :00 each hour) and daily-close (INX/NASDAQ100 at 16:00 ET).  
**Objective:** Maker-first, fee-aware, always-on intraday + EOD machine that preserves capital and compounds a small, repeatable edge.  
**Research sources:** Fee schedule, rulebook & market rules (Kalshi); Indices WS docs (Polygon.io/Massive); relevant CFTC product filings (INX/NASDAQ100). See references at the end.

---

## 0) Thesis & guardrails

- **Where the edge is:** small PMF mispricings + spread capture on liquid bins, especially near **top-of-hour** fixes and **EOD** (16:00 ET).  
- **What kills edge:** taker fees, 1¢ tick rounding on single-lots, stale data near the fix, and over-hedging across correlated bins.  
- **Non‑negotiables:** maker‑first flow, hard **T–2s** pull, freshness gating, and evidence‑based sizing (EV honesty + fill realism).  
- **Capacity goal:** scale quotes to 2–4 best bins per window (SPX/NDX), widen participation as fill model matures.

---

## 1) Market research summary (kept evergreen in repo)

- **Active products:** S&P 500 and Nasdaq‑100 **above/below** (intraday hourly “U” markets) and **range**/above‑below at the **daily close**.  
- **Trading hours:** Kalshi runs continuously with scheduled maintenance windows; individual markets list their own **Last Trading Time** in rules (intraday contracts end at the specified timestamp; EOD at 16:00 ET).  
- **Settlement timing:** typically shortly after expiration; depends on market type and source data availability.  
- **Fees (indices):** documented as `0.035 * C * p * (1-p)` per contract for SPX/NDX markets; **per‑order rounding** to next cent applies. Maker fees may be product‑specific—treat as a configurable switch.  
- **Implication:** at **p≈0.50**, taker fee per 1‑lot rounds to **$0.01**; per 100‑lot ≈ **$0.88** → maker‑only is the default path for hourly bins.

**Keep this section live** via `docs/market_facts.md` regenerated monthly by a script that rechecks fee/rule URLs.

---

## 2) System blueprint (single source of truth)

### Components
1. **Data plane**
   - **Sources:** Polygon Indices WS (real‑time values for SPX/NDX), minute aggregates for robustness; WRDS/CRSP for daily backfills (EOD calibration only).
   - **Ingestion:** resilient WS client with **freshness** metric; rolling minute cache; flat-file snapshots per window under `data/raw/index/YYYYMMDD/HH`.
   - **Quality gates:** clock skew (NTP check), outlier filters, gap-filler (linear & VWAP‑based), “final‑minute strict mode” (<700ms staleness).

2. **Modeling**
   - **PMF generator (Hourly/EOD):** Brownian bridge from *now* → target time with **σ_tod** (time‑of‑day volatility) from last 60 trading days; close model adds **auction variance bump**.
   - **Param store:** versioned JSON under `data/proc/calib/index/{INX,NDX}/{hourly,eod}/YYYY‑WW.json`.
   - **Honesty & sharpness:** reliability curves, Brier score, and **EV‑honesty** per bin; clamp sizing if dishonesty detected.

3. **Execution**
   - **Quoters:** maker‑only, two‑sided with inventory tilt; quote bands derived from PMF skew and fee model; **pull at T–2s**.
   - **Broker:** Kalshi v2 (RSA‑PSS), idempotent order keys, bounded cancel/replace queue, PAL checks, kill‑switch first‑line check.
   - **Fee model:** per‑product config (`indices.taker=0.035`, `indices.maker={0 or 0.0175?}`), per‑order rounding; reimbursement (if any) accounted **only in reporting**, not EV.

4. **Risk**
   - **VaR:** fast Monte Carlo using PMF for current inventory; **per‑family** caps (SPX vs NDX) and **cross‑bin** correlation.
   - **Loss limits:** hard daily/weekly stops; “cool‑off mode” disables quoting but keeps monitoring.

5. **Observability**
   - **Scoreboard:** EV\_after\_fees, realized PnL, fill realism delta, calibration age, quote time‑at‑risk, last freshness.
   - **Runbooks:** `docs/runbooks/{hourly,eod}.md` with checklists (pre‑window, during, post).

6. **Automation**
   - **Scheduler:** US/Eastern aware; knows **on‑the‑hour** windows for INXU/NASDAQ100U and **16:00 ET** for EOD.
   - **State machine:** `IDLE → PREP (T–10m) → MAKE (T–5m to T–2s) → FREEZE (T–2s..T+60s) → POST (ledger & reports)`.
   - **WebSocket sentry:** trip if `freshness_ms > 700` in the final minute → freeze & cancel‑all.

---

## 3) Edge development roadmap

1. **σ_tod calibration**
   - Compute per‑hour realized variances for SPX/NDX; fit smoothing spline; store as params with weekly refresh.
2. **Bridge model**
   - Condition on current price and remaining horizon; add residual clamp from recent realized vs implied; EOD bump calibrated from last 15‑minute variance.
3. **Range ladder composition**
   - Convert PMF on terminal level to bin probabilities (between/above/below); support shared sampling to respect correlations.
4. **Fill model**
   - LOB sim from recorded TOB snapshots; adverse‑selection bias; learn **fill\_prob(price\_offset, time\_to\_fix)** and apply to EV.
5. **Cross‑ladder structuring**
   - Hedge range vs above/below using internal inventory; allocate capital to the better EV after fees.
6. **Honesty enforcement**
   - If per‑bin *EV honesty* < threshold over rolling windows, auto‑downweight or disable that bin.

---

## 4) Backtesting & replay

- **Data:** Polygon minute bars + recorded value ticks; WRDS CRSP for EOD validation.  
- **Simulator:** emits orders at your planned schedule (e.g., T–5m..T–2s), simulates partial fills and cancels; applies **per‑order** fee rounding.  
- **Outputs:** day‑by‑day EV, realized, slippage drift, fill realism, and *ΔEV vs live* parity test.

---

## 5) Pilot protocol (live discipline)

- **Mode:** maker‑only, 1–2 best bins per window, 1–5 lots per quote band, **no taker**.  
- **Gates:** freshness good; σ\_params age < 7d; EV\_after\_fees > X bps; VaR within cap; kill‑switch clear.  
- **Lifecycle:** cancel‑all at **T–2s**; freeze until `settled|cancelled`; reconcile ledgers; update scoreboard.  
- **Scale‑up:** only after 30‑day **EV honesty** and fill realism pass thresholds.

---

## 6) AWS compute plan

- **Workloads:** calibrations, backtests, report generation, and heavy EV sweeps.  
- **Infra:** single container image (`ghcr.io/<org>/kalshi-sys:latest`) with pinned Python + system deps; IaC for batch/spot nodes.  
- **Artifacts:** write to S3 `kalshi-sys-artifacts/{calib,backtests,reports}`; CloudWatch metrics for freshness, quote time‑at‑risk, EV deltas.

---

## 7) Repository structure (target)

```
kalshi-sys/
  src/
    data/                 # WS clients, polygon adapters, freshness sentry
    models/               # PMF/bridge + σ_tod + range composer
    exec/                 # quoters, broker, order queue, PAL
    risk/                 # VaR, inventory, loss caps
    report/               # scoreboard, honesty plots, run summaries
    sched/                # time/holiday aware scheduler & state machine
  tests/
    unit/                 # fee rounding, broker safety, PMF tails
    replay/               # LOB simulator + parity tests
  jobs/                   # calib/backtest/report jobs (AWS)
  configs/
    fees.json             # per-product fee coefficients + maker toggle
    markets.json          # market ids & windows for INX/NDX
  data/                   # proc + raw snapshots (gitignored)
  docs/
    market_facts.md
    runbooks/{hourly,eod}.md
    design/pmf.md
  .env.local.example
```

---

## 8) Concrete task list (checkable)

**Fees & rules**
- [ ] Implement `configs/fees.json` for indices; unit-test rounding (1, 2, 100, 1000 lots).  
- [ ] Script to re-scrape/check fee & rules URLs monthly → update `docs/market_facts.md`.

**Scheduler & sentry**
- [ ] US/Eastern scheduler with holiday calendar and dynamic daylight handling.  
- [ ] Final‑minute **freshness <= 700ms** hard gate; auto‑cancel at **T–2s**.

**PMF & honesty**
- [ ] σ\_tod job with rolling 60d window (SPX/NDX).  
- [ ] Bridge PMF + EOD bump; reliability curves and Brier score plots; clamp sizes by dishonesty.  

**Fill & replay**
- [ ] TOB recorder; LOB sim with adverse selection; learn `fill_prob(..)`; feed into EV.  
- [ ] Replay parity test: ΔEV live vs replay <= ε; break CI if violated.

**Execution & risk**
- [ ] Quoter that uses PMF skew + fees to place maker quotes only in top 2 bins.  
- [ ] VaR monitor + per‑family caps; enforce daily/weekly loss limits in the order path.

**Observability**
- [ ] Scoreboard markdown updated each window; daily rollup; alert on gaps.  
- [ ] Runbooks with pre/during/post checklists; include “no‑go” reasons in reports.

**AWS**
- [ ] Containerize jobs; `jobs/calib_hourly.py` and `jobs/backtest_hourly.py`.  
- [ ] S3 artifact push + CloudWatch metrics for freshness and EV drift.

---

## 9) Prompts & agents (Codex CLI)

- Maintain **AGENTS.md** for roles, guardrails, and workflows.  
- Provide a **single Codex prompt** to: plan → implement → test → document → commit/push → open PR → generate summary.

---

## 10) References (keep updated)

- Kalshi market rules & timelines: help center pages (Timeline & Payout; Rules Summary), CFTC product filings for INX/NDX, and Rulebook “Last Trading Time”.  
- Fees: Fee Schedule PDF for SPX/NDX (0.035 * C * p * (1-p)), note per‑order rounding and maker fee policy.  
- Data: Polygon/Massive Indices WebSocket (real‑time values and minute aggregates).

URLs (for human readers; please verify periodically):
- Timeline & payout: https://help.kalshi.com/markets/markets-101/market-rules/timeline-and-payout
- Rules summary: https://help.kalshi.com/markets/markets-101/market-rules/rules-summary
- Rulebook excerpt (last trading time): https://www.cftc.gov/filings/orgrules/rules1113248695.pdf
- Fees (SPX/NDX): https://kalshi-public-docs.s3.us-east-1.amazonaws.com/fee-schedule.pdf
- Polygon Indices WS: https://polygon.io/docs/websocket/indices/value
- Indices data blog: https://polygon.io/blog/indices-data-has-arrived/
