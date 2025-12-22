# Config Reference

## Metadata
- Generated: 2025-12-22T19:42:20Z
- Git SHA: a907a2eed87531d8178c3dc183d6f070182f9ebe
- Branch: codex/TICKET-000_project_state_rebuild
- Commands: `python tools/project_state_build.py`, `rg --files`, `sed -n '1,200p' README.md`, `sed -n '1,200p' docs/PROGRESS.md`, `sed -n '1,200p' CHANGELOG.md`, `sed -n '1,200p' pyproject.toml`, `sed -n '1,200p' Makefile`

## Core index configs
- `configs/fees.json` — fee coefficients + rounding rules per index series.
- `configs/pal_policy.yaml` — PAL (max loss per strike) for index series.
- `configs/index_var.yaml` — per-family VaR limits.
- `configs/index_correlation.yaml` — correlation-aware VaR and tilt limits.
- `configs/index_ops.yaml` — window timing offsets, cancel buffers, max bins, min EV.
- `configs/pilot.yaml` — pilot caps (1-lot, maker-only, ack required).
- `configs/size_ladder.yaml` — staged sizing ladder for pilot scaling.

## Quality & freshness gates
- `configs/quality_gates.index.yaml` — index-only quality gates + freshness namespace.
- `configs/freshness.index.yaml` — required index feed freshness (Polygon WS).
- `configs/quality_gates.yaml` — macro-oriented gates (still present).
- `configs/freshness.yaml` — macro + index feed freshness thresholds.

## Series & strategy metadata
- `configs/series.yaml` — macro series settlement metadata (CPI/Claims/TENY/Weather).
- `configs/strategies/cpi.yaml` — CPI strategy blend weights and variance config.

## Fee/rule watcher
- `configs/fee_rules_watch.yaml` — URLs for official Kalshi fee schedule + rulebook hashes.

## Systemd / ops templates
- `configs/systemd/*.service` / `configs/systemd/*.timer` — unit/timer templates for supervisors and monitors.
- `configs/logrotate/kalshi-alpha` — logrotate config for runtime logs.
- `configs/cloudwatch/kalshi-supervisor-index.json` — CloudWatch agent config template.

## Deprecated / compatibility
- `configs/fees/series.json` — shim to `configs/fees.json` (deprecated; see file note).

## Notes
- Do not relax pilot constraints or live trading guardrails without explicit approval.
- Index-only scope should be enforced via entrypoints and quality gate scope settings.
