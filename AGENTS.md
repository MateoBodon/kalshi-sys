# AGENTS.md — kalshi-sys (Hourly + Close, SPX/NDX)

## Mission
Maker-only strategies for S&P 500 (INX*/I:SPX) and Nasdaq-100 (NASDAQ100*/I:NDX) **hourly** (INXU/NASDAQ100U) and **close** (INX/NASDAQ100) ladders on Kalshi. Quote small positive-EV bins; measure realized edge honestly; scale only on evidence.

## Non-negotiables
- Maker-only. Never cross. Cancel all quotes by T−2s.
- Hourly targets: 10:00,11:00,12:00,13:00,14:00,15:00,16:00 ET. Close: 16:00 ET.
- Fees: index taker = `round_up(0.035*C*P*(1−P))`; **maker = $0** for these series unless fee schedule changes. Unit tests guard this.
- Quote only if `EV_after_fees ≥ $0.05` and `α·EV − slippage > 0` on paper. Max **1-lots**, **≤2 bins/series**, strict PAL/loss caps.

## Data & secrets
- Polygon Indices WS (minute/second) for I:SPX, I:NDX. Optional SPY/QQQ for health.
- macOS Keychain item: `kalshi-sys:POLYGON_API_KEY`; env fallback; never log secrets.

## Modeling
- Per-hour calibration (σ_now × m_TOD, micro-drift, event-day tails). PIT correction.
- Close ranges integrate PDF over bin and smooth adjacency.

## Execution
- pilot_hourly: open at T−15m, reprice on drift, cancel T−2s.
- Guardrails: kill switch, NO-GO gates, freshness, calibration age.

## Tests & verification
- Fees (goldens incl. 50c → 0.88c), range mass, noon/hourly CDF, PIT; WS reconnect & parquet chunking.
- Smoke: scanners write CSV + Markdown; scoreboard & readiness generate without errors.

## Observability
- JSONL logs; monitors for data latency, α-gap, slippage drift; alert if ledger silent.

## When in doubt
- Small, reversible changes with clear commit messages. Add `TODO(OWNER=mateo)` and proceed with safe defaults.
