# Tests

- `pytest -q` (exit 0)
- `pytest -q` (exit 0, rerun after market-status guard tweak)
- `pytest -q` (exit 0, after freshness monitor market-status guard + CLI helper)
- `PYTHONPATH=src python3 -m kalshi_alpha.exec.collectors.polygon_ws --max-runtime 30` (exit 0; terminated early due to `max_connections`)
- `PYTHONPATH=src python3 -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run --record-tob --telemetry-only --tob-run-id 20251226_074420Z --tob-output-dir data/proc/telemetry --no-ws-listen --now 2025-12-26T19:50:00Z` (exit 0; preflight NO-GO basis_audit_missing:INXU)
- `PYTHONPATH=src python3 -m kalshi_alpha.exec.supervisor_index --series NASDAQ100U --dry-run --record-tob --telemetry-only --tob-run-id 20251226_074517Z --tob-output-dir data/proc/telemetry --no-ws-listen --now 2025-12-26T19:50:00Z` (exit 0; preflight NO-GO basis_audit_missing:NASDAQ100U)
- `PYTHONPATH=src python3 -m kalshi_alpha.exec.reports.telemetry_volume --run-id 20251226_074420Z --report-date 2025-12-26` (exit 0)
- `PYTHONPATH=src python3 -m kalshi_alpha.exec.housekeep --keep-days 10000` (exit 0; removed synthetic old telemetry file)
- `pytest -q` (exit 0, after telemetry proof + bundle artifact updates)
- `pytest -q` (exit 0, after gpt-bundle verification update)
