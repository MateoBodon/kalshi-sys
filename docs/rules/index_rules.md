---
updated: 2025-11-04
tick_size_usd: 0.01
position_limit_usd: 7000000
series:
  INXU:
    display_name: "S&P 500 Noon Above/Below"
    evaluation_time_et: "12:00:00"
    evaluation_clause: "YES if the official S&P 500 index (SPX) is strictly greater than the strike at exactly 12:00:00 ET on the event date."
    timing_clause: "Index prints published on or before 12:00:00 ET are eligible; the first official value stamped at or before 12:00:00 ET is used for settlement."
    fallback_clause: "If no eligible value is published at 12:00 ET, use the most recent official index print prior to that time; if no print is available by 17:00 ET, Kalshi applies the exchange-wide disruption policy (postpone, re-open, or void)."
    reference_source: "S&P Dow Jones Indices LLC, SPX index disseminated via CTA."
    primary_window_et: "Recommended quoting window 11:45–12:00 ET."
  NASDAQ100U:
    display_name: "Nasdaq-100 Noon Above/Below"
    evaluation_time_et: "12:00:00"
    evaluation_clause: "YES if the official Nasdaq-100 index (NDX) is strictly greater than the strike at 12:00:00 ET on the event date."
    timing_clause: "Uses the first official NDX value disseminated on or before 12:00:00 ET."
    fallback_clause: "If no value is available at 12:00 ET, fall back to the most recent official print prior to 12:00; if unavailable through 17:00 ET, the disruption policy applies."
    reference_source: "Nasdaq, Inc., Nasdaq-100 index (NDX) disseminated via UTP."
    primary_window_et: "Recommended quoting window 11:45–12:00 ET."
  INX:
    display_name: "S&P 500 Close Range"
    evaluation_time_et: "16:00:00"
    evaluation_clause: "Resolves on the official S&P 500 index closing print (SPX) at 16:00:00 ET; range contracts pay YES if the closing level lies within the specified bin."
    timing_clause: "Uses the official closing value stamped at or before 16:00:00 ET, including any official corrections disseminated before 17:00 ET."
    fallback_clause: "If the close is delayed, use the latest official value published before 17:00 ET. If no value is published, the disruption policy governs (postpone or void)."
    reference_source: "S&P Dow Jones Indices LLC, official SPX close from CTA."
    primary_window_et: "Recommended quoting window 15:50–16:00 ET."
  NASDAQ100:
    display_name: "Nasdaq-100 Close Range"
    evaluation_time_et: "16:00:00"
    evaluation_clause: "Resolves on the official Nasdaq-100 index (NDX) closing level at 16:00:00 ET; YES if the closing print lands within the ladder bin."
    timing_clause: "Takes the first official NDX value published at or before 16:00:00 ET, including same-day official corrections."
    fallback_clause: "If no closing value is published at 16:00 ET, use the latest official value before 17:00 ET; otherwise follow the disruption procedure."
    reference_source: "Nasdaq, Inc., official Nasdaq-100 index close."
    primary_window_et: "Recommended quoting window 15:50–16:00 ET."
---

# Index Ladder Rule Summary — INX / INXU / NASDAQ100 / NASDAQ100U

This document condenses the key settlement semantics from Kalshi’s rule PDFs for the four Polygon-powered index ladder series.  
All times are **Eastern Time (ET)**, payouts are in USD, and tickets trade in $1 notional increments with a **$0.01 minimum tick**.  
The exchange-wide **position limit** is **$7,000,000 notional** per member across contracts that reference these indices.

## Core Settlement Clauses

| Series | Observation | On/Before Timing | Fallback (no print) | Primary Source |
| --- | --- | --- | --- | --- |
| INXU | SPX level at 12:00:00 ET | First official SPX print on/before 12:00:00 | Use latest official SPX value prior to 12:00; disruption policy if no print by 17:00 | S&P Dow Jones Indices |
| NASDAQ100U | NDX level at 12:00:00 ET | First official NDX print on/before 12:00:00 | Use latest official NDX value prior to 12:00; disruption policy if unavailable | Nasdaq |
| INX | SPX closing level (16:00:00 ET) | First official SPX close on/before 16:00:00 | Latest official print before 17:00; disruption policy otherwise | S&P Dow Jones Indices |
| NASDAQ100 | NDX closing level (16:00:00 ET) | First official NDX close on/before 16:00:00 | Latest official print before 17:00; disruption policy otherwise | Nasdaq |

> “On/before 12:00 p.m. (ET); if no print at timestamp, use most recent prior.” — Kalshi Index Ladder Rules (Noon ladders, Oct 1 2025)
>
> “On/before 4:00 p.m. (ET); if no print at timestamp, use most recent prior.” — Kalshi Index Ladder Rules (Close ladders, Oct 1 2025)

> “Tick size: $0.01.” — Kalshi Index Ladder Rules (Oct 1 2025)

> “Position limit: $7,000,000 notional across member accounts.” — Kalshi Index Ladder Rules (Oct 1 2025)

- **Disruption policy:** Mirrors Kalshi’s standard ladder procedures—Kalshi may postpone settlement, seek third-party confirmation, or void if the primary source fails to publish.
- **Corrections:** Same-day official index corrections prior to 17:00 ET supersede preliminary prints.
- **On/before guidance:** If the exact 12:00:00 ET or 16:00:00 ET print is absent, settlement uses the most recent prior official index value disseminated before the timestamp. <!-- Source: Kalshi Index Ladder Rules (Oct 1 2025), §2 -->
- **Tick size:** $0.01 price increments, implying 1¢ minimum P&L per contract.
- **Position limit:** $7MM notional per member, per exchange rule filings (enforced via internal caps well below the exchange maximum).

## Maker vs Taker Fees

- **Indices exception:** Makers in `INX*` / `NASDAQ100*` pay $0.00 per contract; takers owe `0.035 × contracts × price × (1 − price)`, rounded up to the nearest cent.
- **Reference:** See `docs/kalshi-fee-schedule.pdf` (effective Oct 1 2025) for the full fee schedule and supporting rate table.

## Operational Windows

- **Noon ladders (INXU / NASDAQ100U):** Recommended quoting window **11:45–12:00 ET**. Ensure Polygon minute feed latency is <30 seconds and websocket depth is healthy before quoting.
- **Close ladders (INX / NASDAQ100):** Recommended quoting window **15:50–16:00 ET**. Require live Polygon minute bars with <20-second latency and closing auction sanity checks.
- **Cancel buffer:** `configs/index_ops.yaml` holds the authoritative `cancel_buffer_seconds = 2`; scanners and microlive cancel-all at T−2 s and surface the value as `ops_cancel_buffer_seconds` in monitor payloads.

## Data Integrity Checklist

1. Confirm Polygon API key loads via macOS Keychain label `kalshi-sys:POLYGON_API_KEY` or `POLYGON_API_KEY` environment variable.
2. Snapshot archive must include the latest minute bars and websocket aggregates for both `I:SPX` and `I:NDX` (fallback equities: `SPY`, `QQQ`).
3. Alerts: halt quoting if kill-switch present, websocket last message age ≥2 seconds, or ET clock drift exceeds 1 second.
4. Verify monitor metadata: `ops_timezone` should read `America/New_York`, `ops_target_et`/`ops_target_unix` trace the target resolution, and `data_timestamp_used` reflects the orderbook snapshot time. If the `'on/before'` fallback fires, `ops_target_fallback` appears as `"on_before"`—log the rationale in the ops notes.

> Always defer to the latest official Kalshi rule PDFs for definitive language. This summary is operational guidance for the index ladder pipeline.
