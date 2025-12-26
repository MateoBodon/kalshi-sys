# Agent Run

Goal: Ticket #6 — Fix index CLI smoke entrypoints (preflight_index + supervisor_index).

Summary:
- Added preflight_index CLI summary + go/no-go artifact writer.
- Added supervisor_index preflight summary + artifact write on NO-GO.
- Added stdout fixture tests for both CLIs.
- Updated PROGRESS + CHANGELOG.

Commands:
- rg -n "def main\\(|argparse|preflight_index|go_no_go\\.json|go_no_go" -S src/kalshi_alpha/exec
- pytest -q
- python -m kalshi_alpha.exec.preflight_index (fails: missing module path)
- PYTHONPATH=src python -m kalshi_alpha.exec.preflight_index
- python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run (fails: missing module path)
- PYTHONPATH=src python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run
- make gpt-bundle TICKET=TICKET-006_index_cli_smoke RUN_NAME="20251222_002818Z_TICKET-006_index_cli_smoke"

Tests:
- pytest -q

Artifacts:
- reports/_artifacts/go_no_go.json (updated)
