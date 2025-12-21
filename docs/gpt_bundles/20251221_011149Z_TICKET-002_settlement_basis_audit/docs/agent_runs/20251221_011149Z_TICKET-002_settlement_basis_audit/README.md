# Agent Run README

Goal: implement the settlement basis audit tool for index ladder windows (Polygon vs Kalshi expiration value) with reproducible outputs and tests.

Summary:
- Added `tools/settlement_basis_audit.py` to generate daily basis datasets + markdown reports with flip-risk flags.
- Added Kalshi public client event-detail fetch for settlement values and offline fixtures for Polygon/Kalshi.
- Added unit test for offline fixtures and fixed pytest subprocess PYTHONPATH for backtest CLI tests.
- Generated offline reports/datasets for INXU and NASDAQ100U on 2025-11-10.

Commands:
- `pytest -q`
- `python tools/settlement_basis_audit.py --day 2025-11-10 --series INXU --offline-fixtures`
- `python tools/settlement_basis_audit.py --day 2025-11-10 --series NASDAQ100U --offline-fixtures`
- `make gpt-bundle TICKET=TICKET-002_settlement_basis_audit RUN_NAME=20251221_011149Z_TICKET-002_settlement_basis_audit`

Tests:
- `pytest -q`

Artifacts:
- `reports/settlement_basis/2025-11-10_INXU.md`
- `reports/settlement_basis/2025-11-10_NASDAQ100U.md`
- `data/proc/settlement_basis/2025-11-10_INXU.parquet`
- `data/proc/settlement_basis/2025-11-10_NASDAQ100U.parquet`
- `docs/gpt_bundles/gpt_bundle_TICKET-002_settlement_basis_audit_20251221_011149Z_TICKET-002_settlement_basis_audit.zip`

Risks / Notes:
- The tool fails closed if the Kalshi event payload lacks an expiration/settlement value field; confirm the live field name before relying on online runs.
- Offline fixtures cover a subset of hourly windows; missing windows show null values in the report summary.
