# Tests

- `pytest -q` (pass; 124 passed, 746 skipped)
- `python3 -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run --offline --now 2025-12-30T10:50:00-05:00` (pass; dry-run/offline smoke)
- Rerun (post-status update): `pytest -q` (pass; 124 passed, 746 skipped)
- Rerun (post-status update): `python3 -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run --offline --now 2025-12-30T10:50:00-05:00` (pass; dry-run/offline smoke)
