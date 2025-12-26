# Run README

Goal: Decouple index-only GO/NO-GO from macro freshness and add explicit scope in artifacts for index ladders.

Summary:
- Added explicit scope metadata to freshness feeds and set index-only freshness defaults.
- Added scoped filtering to quality gates and scoped GO/NO-GO artifacts for index runs (preflight + scan).
- Added scoped preflight regression test (macro stale, index fresh) and index artifact scope assertions; changelog entry updated.

Commands:
- pytest -q
- make monitors (failed: /bin/sh: python not found)
- PYTHON=python3 make monitors
- python3 -m kalshi_alpha.exec.preflight_index --offline
- python3 -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run --offline --max-runtime-seconds 5

Tests:
- pytest -q (117 passed, 741 skipped)
- make monitors (failed: /bin/sh: python not found)
- PYTHON=python3 make monitors (OK)
- python3 -m kalshi_alpha.exec.preflight_index --offline (exit 1; NO-GO)
- python3 -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run --offline --max-runtime-seconds 5 (exit 0; preflight NO-GO)

Artifacts:
- reports/_artifacts/go_no_go.json (preflight/supervisor runs)
- reports/_artifacts/monitors/*.json (make monitors)
