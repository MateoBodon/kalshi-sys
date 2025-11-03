
# VISION.md — SPX/NDX Index Ladders (Noon & Close), Polygon-Powered

**Repo:** `kalshi-sys`  
**Focus:** Maker-only strategies for **S&P 500 (I:SPX)** and **Nasdaq‑100 (I:NDX)** **noon above/below** (INXU / NASDAQ100U) and **close ranges** (INX / NASDAQ100) on Kalshi.  
**Example market IDs (Nov 3, 2025):**
- Noon SPX: `KXINXU-25NOV03H1200`, Market: `KXINXU-25NOV03H1200-T6844.9999`
- Noon NDX: `KXNASDAQ100U-25NOV03H1200`, Market: `KXNASDAQ100U-25NOV03H1200-T25959.99`
- Close SPX: `KXINX-25NOV03H1600`, Market: `KXINX-25NOV03H1600-B6862`
- Close NDX: `KXNASDAQ100-25NOV03H1600`, Market: `KXNASDAQ100-25NOV03H1600-B25950`

---

## 0) Executive Summary

- **Goal:** Consistently extract small, repeatable edge by posting **maker** quotes in **two daily windows**: **11:45–12:00 ET** (noon fix) and **15:50–16:00 ET** (close).  
- **Data spine:** **Polygon Indices** (`I:SPX`, `I:NDX`) for minute/second aggregates + WebSocket aggregates; optional **SPY/QQQ** from Stocks tier for sanity/fallback.  
- **Method:** For a short horizon τ, model the probability that the index **is above a strike at 12:00** (noon AB/BL) or **finishes in a range at 16:00** (close ranges). Use **intraday realized volatility** and **time‑of‑day seasonality**, a small **micro‑drift**, and **tail inflation** on event days.  
- **Execution:** Maker‑only, 1‑lots, ≤ 2 bins per series, **EV_after_fees ≥ $0.05**, **truncated Kelly** sizing with hard **PAL** caps, auto‑cancel at T−1–2s.  
- **Graduation:** Paper → **readiness thresholds** → tiny live pilot. Culture of **alpha honesty** (fills/slippage modeled and measured).

---

## 1) Markets, Rules, and Ops Windows

- **Series:** `INXU` / `NASDAQ100U` (noon **above/below**), `INX` / `NASDAQ100` (close **range**/AB/BL variants).  
- **Truth semantics (per market page Rules):** `<on/before>` + `<time>` + `<date>` in **ET**; noon uses 12:00:00 ET; close uses 16:00:00 ET *official index print*. If there is no data at the timestamp, use the most recent value prior to that time.  
- **Tick:** $0.01. **Position limit:** $7,000,000 notional per member on $1 contracts (we enforce much lower internal caps).  
- **Ops windows:** quote **11:45–12:00 ET** and **15:50–16:00 ET**; avoid last‑seconds thrash unless edge is extreme; obey Kalshi maintenance windows.

> Implementation detail: cache the per‑series **Rules Summary** (source/time/“on or before”) in `docs/rules/index_rules.md` and load at runtime to configure the scanner precisely.

---

## 2) Data, Secrets, and Storage

### 2.1 Polygon indices
- **Symbols:** `I:SPX`, `I:NDX`; sanity/fallback: `SPY`, `QQQ`.  
- **REST aggregates:** minute/second historical bars with automatic **range chunking** and pagination stitching.  
- **WebSocket aggregates:** subscribe only during pre‑windows (11:40–12:01, 15:45–16:01) to minimize traffic.

### 2.2 Secrets (macOS Keychain first)
- Item: `kalshi-sys:POLYGON_API_KEY` in the **login** keychain; fallback env `POLYGON_API_KEY`.  
- Loader precedence: Env → Keychain → **fatal error**. Never log or commit secrets.

### 2.3 Layout
- Raw → `data/raw/polygon/{symbol}/YYYY-MM-DD.parquet` (minute).  
- Proc → `data/proc/calib/index/{spx,ndx}/{noon,close}/params.json` (σ_TOD, mixture/tail, PIT).  
- Reports → `reports/index_ladders/{run_id}/…` (per‑bin p_yes, EV after fees, ledger diffs, plots).  
- Ledger → `data/proc/ledger_all.parquet` (paper & live).

---

## 3) Modeling (Noon & Close)

### 3.1 Features
- `S_t` (level), `d = strike − S_t` (AB/BL) or `[a,b]` range (close).  
- `σ_now`: realized vol from last 30–60 minutes of 1‑sec/min returns with EWMA smoothing.  
- `m_TOD(τ)`: **time‑of‑day seasonal** multiplier for horizon τ (noon vs close curves).  
- Micro‑drift from last 5–10 minutes (bounded).  
- Event masks for **FOMC/CPI** days → tail inflation factors.  
- Sanity checks via SPY/QQQ spread/impact (health only).

### 3.2 Probability maps
- **Noon AB/BL:** Short‑horizon lognormal base.  
  - `Δx = log(S_T/S_t) ~ N(μ_t τ, σ_t^2 τ)` with `σ_t = σ_now × m_TOD(τ)` and small `μ_t`.  
  - `q_above = 1 − Φ( (ln(K/S_t) − μ_t τ) / (σ_t √τ) )`.  
  - **PIT calibration** (isotonic or Platt) to correct residual bias.  
- **Close ranges:** integrate PDF across each `[a,b]`; add **adjacent‑bin smoothing** to avoid EV cliffs after fees.

### 3.3 Calibration regime
- Pull 6–12 months of minute bars for `I:SPX`, `I:NDX`.  
- Fit `m_TOD(τ)`; fit residual variance bumps for late‑day; flag event‑day tails.  
- Evaluate **Brier**, **CRPS**, **reliability**; store PIT mapping per series/horizon.  
- Recalibrate ≤ 14 days; checksum JSON params.

---

## 4) Fees, EV, Fill, Slippage

### 4.1 EV after fees
- For YES at price `p` with calibrated probability `q` and series fee function `f_series(p)` (USD/contract):
  - `EV_yes = q * (1 − f_series(p)) − (1 − q) * f_series(p)`  
  - `EV_no  = (1 − q)*(1 − f_series(1−p)) − q * f_series(1−p)`  
- **Series‑specific fee curve:** parse official fee tables for INX/INXU/NASDAQ100/NASDAQ100U into JSON; unit‑test against published table rows.  
- Scanner threshold: **EV_after_fees ≥ $0.05** (configurable).

### 4.2 Fill probability (α) & slippage
- α: probability our resting maker quote is lifted before reprice; model vs spread, depth, and τ.  
- Slippage: expected loss vs mid when lifted; series‑specific curve (SPX vs NDX).  
- **Honesty metric:** `Realized_Δbps ≈ EV_after_fees × α − slippage` must remain positive on paper before any live.

---

## 5) Execution Policy & Risk

- **Maker‑only**, default **1‑lots**, **≤ 2 bins per series**.  
- Sizing: truncated **Kelly** on expected Δbps with **per‑bin PAL** and **per‑series cap**; never exceed internal caps.  
- Quote control: best price ±1 tick (favorability) while preserving edge after fees.  
- Cancel/repost: on price drift, α saturation, or end‑of‑window; **cancel by T−1–2s**.  
- **Halt conditions:** stale data, monitor warnings, kill‑switch set, or GO/NO‑GO = NO.

---

## 6) Quality Gates & Readiness

- **Freshness:** Polygon bar latency ≤ threshold; SPX/NDX clock aligned to ET; SPY/QQQ sanity deltas within bounds.  
- **Calibration:** params age ≤ 14 days; event masks applied.  
- **Paper performance (min to graduate):** ≥ 300 fills over last 14 days; mean Δbps ≥ 6; **t‑stat ≥ 2**; small α‑gap.  
- **Reports:** scoreboard (7d/30d), pilot_readiness; GO/NO‑GO rationale archived.

---

## 7) Pipelines (CLI)

### 7.1 Ingest
```
python -m kalshi_alpha.drivers.polygon_index.client --symbols I:SPX I:NDX --start 2025-06-01 --end 2025-11-03 --write-parquet
```

### 7.2 Calibrate
```
python -m kalshi_alpha.jobs.calibrate_noon  --symbols I:SPX I:NDX --months 12
python -m kalshi_alpha.jobs.calibrate_close --symbols I:SPX I:NDX --months 12
```

### 7.3 Scan (paper)
```
python -m kalshi_alpha.exec.scanners.scan_index_hourly  --offline --report
python -m kalshi_alpha.exec.scanners.scan_index_close --offline --report
```

### 7.4 Verify & readiness
```
python -m kalshi_alpha.exec.scoreboard --window 7 --window 30
python -m kalshi_alpha.exec.pilot_readiness
```

### 7.5 Pilot (human‑gated)
```
python -m kalshi_alpha.exec.pipelines.preflight --mode index_noon
python -m kalshi_alpha.exec.runners.pilot --series INXU --broker live --i-understand-the-risks --kill-switch-file data/proc/state/kill_switch
```

---

## 8) Code Organization (deltas)

```
src/kalshi_alpha/
  drivers/polygon_index/{client.py,symbols.py,snapshots.py}
  strategies/index/{hourly_above_below.py,close_range.py,cdf.py,params.py}
  core/{fees.py, ev.py, kelly.py, slippage.py, fills.py}
  exec/scanners/{scan_index_hourly.py, scan_index_close.py}
  jobs/{calibrate_noon.py, calibrate_close.py}
  utils/{keys.py, tz.py}
data/{raw,proc,proc/calib/index,...}
reports/index_ladders/<run_id>/*
docs/{RUNBOOK.md, rules/index_rules.md}
tests/{test_fees_inx.py, test_keys.py, test_polygon_client.py, test_noon_math.py, test_range_mass.py, ...}
```

---

## 9) Testing & CI

- **CI:** `ruff`, `mypy`, `pytest -q` must pass.  
- **Unit tests:**  
  - Keychain loader (no secrets in logs).  
  - Polygon REST chunking + WS reconnect.  
  - Fee JSON lookup for INX/NDX (golden rows).  
  - Math: Φ / CDF, noon pmap, range integrals.  
- **Backtests:** reliability curves & CRPS snapshots regenerated on calibration.  
- **Smoke:** DRY scans produce Markdown + CSV reports; telemetry redacts secrets.

---

## 10) Telemetry & Ops

- Structured JSONL logs; daily rotation; ORDER/QUOTE lifecycle events redacted.  
- Monitors: data freshness; fills vs α; slippage drift vs model; readiness deltas.  
- Kill‑switch: file flag; broker cancel‑all intent; runner respects immediately.

---

## 11) Roadmap (“what will add more edge later”)

- **Option IVs** (upgrade tier): SPX/NDX implied vol to refine σ_now.  
- **Futures micro** (ES/NQ) if accessible.  
- **Closing imbalance inference** from price/vol into 15:50–16:00.  
- **Bandit bin‑selection** under caps.  
- **Cross‑series risk** optimization.

---

## 12) Definition of Done (Pilot)

- Paper **fills ≥ 300/14d**, **Δbps ≥ 6**, **t ≥ 2**, freshness OK, calib ≤14d old, kill‑switch hot.  
- Pilot **GO**; live is 1‑lot maker, ≤ 2 bins/series; cancel at boundary; archive reports.  
- Scale size *only if* realized Δbps ≥ threshold after slippage on live.

---

### Appendix A — Keychain Helper (macOS, Python)
```python
# src/kalshi_alpha/utils/keys.py
import os, subprocess, getpass

def load_polygon_key() -> str:
    env = os.getenv("POLYGON_API_KEY")
    if env:
        return env.strip()
    cmd = [
        "security","find-generic-password",
        "-a", getpass.getuser(),
        "-s", "kalshi-sys:POLYGON_API_KEY",
        "-w"
    ]
    out = subprocess.run(cmd, capture_output=True, check=True, text=True)
    return out.stdout.strip()
```

### Appendix B — Fee Adapter (placeholder)
```python
# src/kalshi_alpha/core/fees.py
def fee_inx_series(p: float) -> float:
    """Return USD fee for 1 contract at probability p for INX/INXU/NASDAQ100/NASDAQ100U.
    Implement via JSON look-up of the official fee table for these series.
    """
    ...
```

### Appendix C — Hourly Probability (sketch)
```python
# strategies/index/hourly_above_below.py
def p_above_noon(S_t, K, tau_min, sigma_now, m_tod, mu=0.0):
    import math
    from .cdf import norm_cdf
    tau = max(tau_min, 1e-9) / 1440.0  # minutes → days (or seconds → fraction of day)
    sigma = max(1e-9, sigma_now * m_tod)
    z = (math.log(K / S_t) - mu * tau) / (sigma * math.sqrt(tau))
    return 1.0 - norm_cdf(z)
```

---

**Single source of truth:** If it isn’t reflected here (or in updated RUNBOOK/tests), it isn’t real.
