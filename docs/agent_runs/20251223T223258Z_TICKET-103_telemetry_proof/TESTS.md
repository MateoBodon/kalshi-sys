# Tests

- pytest -q (exit 0)
  - results: 124 passed, 736 skipped
- pytest -q (exit 0)
  - results: 124 passed, 736 skipped
- PYTHONPATH=src python3 -m kalshi_alpha.exec.collectors.polygon_ws --max-runtime 15 (exit 0; interrupted)
- PYTHONPATH=src python3 -m kalshi_alpha.exec.supervisor_index --series NASDAQ100 --dry-run --record-tob --telemetry-only --tob-run-id 20251223_225626Z --tob-output-dir data/proc/telemetry --now 2025-12-23T15:55:00-05:00 (exit 0; NO-GO + WS stale)
- PYTHONPATH=src python3 -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run --record-tob --telemetry-only --tob-run-id 20251223_225859Z --tob-output-dir data/proc/telemetry --now 2025-12-23T14:55:00-05:00 --no-ws-listen (exit 0; telemetry artifacts created)
- PYTHONPATH=src python3 -m kalshi_alpha.exec.reports.telemetry_volume --run-id 20251223_225859Z (exit 0)
- PYTHONPATH=src python3 -m kalshi_alpha.exec.housekeep --keep-days 30 (exit 0; removed 27 artifacts)
