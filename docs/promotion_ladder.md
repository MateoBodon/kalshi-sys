# Promotion Ladder — kalshi-sys

## Stage 0 → Stage 1: Simulation Ready
- **Scoreboard**: 7d and 30d windows populated; EV honesty clamps ≥0.85 and ΔEV within ±10 bps.
- **SLOs**: Freshness p95 ≤400 ms and time-at-risk p95 ≥120 s for the primary series; VaR headroom ≥75% of cap.
- **Monitors**: fee/rule watcher OK, sigma-drift shrink = 1.0, drawdown + kill-switch green.
- **Artifacts**: `report/digest.py --date yesterday --write` run, digest uploaded, and linked in REPORT.md.
- **Runbooks**: hourly/eod checklists completed with proof of telemetry uploads.

## Stage 1 → Stage 2: Pilot (1–2 bins)
- **Maker-only discipline**: proposals limited to ≤2 bins, PAL policies configured, daily/weekly stops active.
- **Fill realism**: fill−α gap within ±2 pp over last 7d, modeled α shrink < early warning.
- **Sigma drift**: shrink ≤0.85 with action plan; sigma monitor artifact green with last ack <7d.
- **Fee watcher**: latest checksums acknowledged; `reports/_artifacts/monitors/fee_rules.json` status OK.
- **Docs**: digest + scoreboard attached to PR/REPORT, promotion rationale logged in this ladder.

## Stage 2 → Stage 3: Production (scaling lots)
- **SLOs**: EV honesty Δbps between −5/+5, time-at-risk p95 ≥180 s, freshness p95 ≤300 ms across INXU/NASDAQ100U & closes.
- **Risk**: correlation-aware VaR + portfolio manager green; weekly drawdown telemetry ≤25% of cap for trailing 4 weeks.
- **Automation**: CloudWatch publishing enabled (`python -m kalshi_alpha.exec.scoreboard --publish-slo-cloudwatch`), daily digest pushed to S3, fee/sigma monitors in CI.
- **Runbooks & postmortems**: no open NO-GO reasons; last two incidents documented via `docs/postmortem_template.md`.
- **Approvals**: ops + risk sign-off recorded in REPORT.md with links to scoreboard/digest artifacts and monitor JSON.
- **Allocator & regimes**: Range/AB structure sigma, regime multipliers, and microprice throttle stats included in daily digest + scoreboard; allocator budgets signed off in REPORT.md.
