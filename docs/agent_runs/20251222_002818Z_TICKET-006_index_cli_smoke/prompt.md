You are Codex running in Codex CLI inside the kalshi-sys repo.

READ FIRST (binding):
- Read AGENTS.md and follow it exactly.
- Default posture must remain PAPER/dry-run. Do NOT make live trading possible by default.
- Do NOT add/print/commit secrets (API keys, PEMs, webhooks).
- Do NOT commit docs/agent_runs/* or docs/gpt_bundles/* (they are gitignored). Still create them locally and include them in the review bundle.
- Work in small commits. EACH commit body must include: `Tests: <exact commands you ran>`.

Ticket to complete: Ticket #6 — Fix index CLI smoke entrypoints (preflight_index + supervisor_index)

Acceptance criteria (must all be true):
1) `python -m kalshi_alpha.exec.preflight_index`:
   - runs without silent exit (even if NO-GO),
   - prints a single-line verdict (GO/NO-GO) + reasons count,
   - writes/updates `reports/_artifacts/go_no_go.json` (or the repo’s canonical path used elsewhere).
2) `python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run`:
   - prints a single-line verdict (GO/NO-GO) + reasons count for the preflight,
   - writes/updates the same `go_no_go.json` artifact.
3) Add at least one fixture-based test asserting stdout is non-empty and includes the verdict line(s), without requiring any live keys.
4) Update docs/PROGRESS.md + CHANGELOG.md with a Ticket #6 entry (Gate status remains PAPER).
5) Create a complete run log under docs/agent_runs/<RUN_NAME>/ and finish by generating a new GPT bundle:
   - `make gpt-bundle TICKET=TICKET-006_index_cli_smoke RUN_NAME="<RUN_NAME>"`

Process (do NOT write a long plan upfront — act):
1) Explore:
   - Open/read:
     - AGENTS.md
     - docs/CODEX_SPRINT_TICKETS.md (Ticket #6 definition)
     - docs/PLAN_OF_RECORD.md (Gate A expectations)
   - Locate current CLIs and artifact writers:
     - `rg -n "def main\\(|argparse|preflight_index|go_no_go\\.json|go_no_go" -S src/kalshi_alpha/exec`
     - Inspect:
       - src/kalshi_alpha/exec/preflight_index.py
       - src/kalshi_alpha/exec/supervisor_index.py
       - any helper that writes go/no-go artifacts (gate utils / reports / monitors)
   - Find existing tests covering preflight/supervisor output and extend them.

2) Implement (small, safe, testable):
   - In preflight_index:
     - Ensure main() always prints ONE line like:
       - `PRECHECK index: GO reasons=0 series=INXU,NASDAQ100U,...`
       - or `PRECHECK index: NO-GO reasons=3 series=INXU ...`
     - Ensure it writes the canonical `reports/_artifacts/go_no_go.json` (match what supervisor/scoreboard expects).
   - In supervisor_index:
     - After preflight runs (or is evaluated), print ONE line summary:
       - `SUPERVISOR preflight: GO reasons=0 series=INXU (broker=dry)`
       - or `SUPERVISOR preflight: NO-GO reasons=...`
     - Ensure the go/no-go artifact is written/updated even on NO-GO (so on-call can rely on it).
   - Do NOT relax any safety gates. Do NOT add live behavior. Prefer fail-closed.

3) Tests:
   - Run `pytest -q`.
   - Add/extend a test that captures stdout for:
     - preflight_index main()
     - supervisor_index invoked in offline/dry mode (use existing offline flags/fixtures; do not require keys)
   - Record results in docs/agent_runs/<RUN_NAME>/TESTS.md and commands.log.
   - If `python -m ...` requires PYTHONPATH=src in this environment, record both:
     - the failing command, and
     - the corrected invocation (but do not “solve” by hiding the failure; document it).

4) Document:
   - Update docs/PROGRESS.md with Ticket #6 status + evidence (tests + run log path).
   - Update CHANGELOG.md with a concise Ticket #6 entry.
   - If you discover the canonical artifact path differs from reports/_artifacts/go_no_go.json, update runbooks/PLAN_OF_RECORD references to match (minimal edit, no scope creep).

5) Branching + run logs + bundle:
   - Create branch: `git checkout -b codex/TICKET-006_index_cli_smoke`
   - Set RUN_NAME:
     - RUN_NAME="$(date -u +"%Y%m%d_%H%M%SZ")_TICKET-006_index_cli_smoke"
   - Create run log dir:
     - mkdir -p "docs/agent_runs/$RUN_NAME"
   - Write required run log files (per AGENTS.md):
     - docs/agent_runs/$RUN_NAME/prompt.md  (paste this prompt)
     - docs/agent_runs/$RUN_NAME/commands.log
     - docs/agent_runs/$RUN_NAME/TESTS.md
     - docs/agent_runs/$RUN_NAME/RESULTS.md
     - docs/agent_runs/$RUN_NAME/META.json  (include start/end UTC, branch, web_search_used)
     - docs/agent_runs/$RUN_NAME/artifacts.json
     - docs/agent_runs/$RUN_NAME/diff.patch  (git diff or git show)
   - Secret scan the staged diff before commit:
     - `git diff --cached | rg -n "API_KEY|PRIVATE_KEY|Bearer|BEGIN RSA|webhook" || true`
   - Commit in small logical commits; include tests in commit body.
   - Generate review bundle and record its path in RESULTS.md:
     - `make gpt-bundle TICKET=TICKET-006_index_cli_smoke RUN_NAME="$RUN_NAME"`

Finish only when:
- acceptance criteria are met,
- tests are passing and recorded,
- PROGRESS + CHANGELOG updated,
- bundle generated successfully.
