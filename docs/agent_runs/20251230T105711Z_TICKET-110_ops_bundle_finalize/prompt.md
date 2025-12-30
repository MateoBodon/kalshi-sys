You are operating under AGENTS.md (binding). Do NOT write a long upfront plan. Follow: explore → implement → test → document.

Ticket: TICKET-110 — Finalize AWS PAPER supervisor artifacts (commit + bundle completeness)

GOAL:
Make the AWS PAPER supervisor deployment reproducible + reviewable by:
1) ensuring the systemd unit file is committed in-repo,
2) ensuring gpt-bundles include the critical deploy/config/runbook files for audit,
3) cleaning up ticket statuses (TICKET-106 should be FAIL until this is done),
4) producing a new bundle for review.

CONSTRAINTS / STOP-THE-LINE:
- Do not make live trading possible by default. Default posture must remain PAPER/dry-run.
- Never log secrets. Never print env var values, tokens, keys, auth headers.
- No “fake fixes” (don’t disable the supervisor or hardcode outputs). We want real auditability.

WORKFLOW REQUIREMENTS (from AGENTS.md):
- Create a feature branch: codex/TICKET-110_ops_bundle_finalize
- Make 2–6 small commits. Each commit body MUST include: "Tests: <commands>"
- End with a clean git status.

STEP 1 — EXPLORE
- Read: AGENTS.md, docs/CODEX_SPRINT_TICKETS.md, docs/PROGRESS.md
- Inspect current repo state:
  - ls configs/systemd and configs/cloudwatch
  - confirm whether configs/systemd/kalshi-index-supervisor-paper.service exists and is tracked by git
  - inspect tools/gpt_bundle_builder.py to see what is currently staged into bundles
  - confirm which runbooks were updated for AWS supervisor (docs/runbooks/aws_supervisor_index.md, docs/runbooks/oncall_checks.md)

STEP 2 — IMPLEMENT (make repo reproducible + bundle reviewable)
A) Systemd unit file (must be committed)
- Ensure configs/systemd/kalshi-index-supervisor-paper.service exists in-repo and is committed.
- The unit MUST be paper-only. Require hard args like:
  - --dry-run
  - explicit --series INXU NASDAQ100U INX NASDAQ100 (or equivalent strict scope)
  - set a heartbeat cadence (e.g., --heartbeat-seconds 60) if supported
- Ensure unit uses a non-root service user (e.g., kalshi) and an EnvironmentFile path that is NOT committed (e.g., /etc/kalshi/kalshi-supervisor.env).
- Add/adjust comments in the unit to make the “paper-only” posture obvious.

B) Bundle completeness for ops deploy review
- Update tools/gpt_bundle_builder.py staging so the gpt bundle includes:
  - configs/systemd/kalshi-index-supervisor-paper.service
  - configs/cloudwatch/kalshi-supervisor-index.json
  - docs/runbooks/aws_supervisor_index.md
  - docs/runbooks/oncall_checks.md
  - reports/ops/aws_supervisor_dryrun_2025-12-30.md (already staged by recent change; keep it)
- Keep the bundle minimal: include only these targeted files/patterns (don’t zip the entire repo).

C) Update ticket status docs
- In docs/CODEX_SPRINT_TICKETS.md:
  - mark TICKET-106 as FAIL with a one-line reason (missing committed unit/bundle reviewability)
  - add a new TICKET-110 entry at the bottom with the acceptance criteria above
- In docs/PROGRESS.md: record TICKET-106 FAIL + TICKET-110 in progress.
- Update CHANGELOG.md for TICKET-110 once complete.

STEP 3 — TEST
- Run: pytest -q
- Run a local paper smoke to validate CLI wiring (no secrets):
  - python3 -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run --offline --now 2025-12-30T10:50:00-05:00
  (If args differ, use the correct paper-safe equivalent.)
- Ensure git status is clean after commits.

STEP 4 — DOCUMENT (run log + reproducibility proof)
- Create run log dir: docs/agent_runs/<RUN_NAME_NEXT>/ where RUN_NAME_NEXT = $(date -u +%Y%m%dT%H%M%SZ)_TICKET-110_ops_bundle_finalize
- Write/Update required run log files:
  - prompt.md (this prompt)
  - RUN.md (what changed + why)
  - NOTES.md (what you inspected; key findings)
  - COMMANDS.md (commands + exit codes)
  - TESTS.md (tests run + results)
  - FILES_TOUCHED.md
  - ARTIFACTS.md (list the bundle + any reports)
  - META.json (minimal metadata: run_name, ticket, branch, timestamp_utc)

STEP 5 — BUNDLE FOR REVIEW (required)
- Generate bundle:
  - PYTHON=python3 make gpt-bundle TICKET=TICKET-110 RUN_NAME=<RUN_NAME_NEXT>
- Verify the zip contains the critical review files:
  - unzip -l docs/gpt_bundles/gpt_bundle_TICKET-110_*.zip | rg "configs/systemd/kalshi-index-supervisor-paper.service|configs/cloudwatch/kalshi-supervisor-index.json|docs/runbooks/aws_supervisor_index.md|docs/runbooks/oncall_checks.md|reports/ops/aws_supervisor_dryrun_2025-12-30.md"
- Record the final bundle path in:
  - docs/agent_runs/<RUN_NAME_NEXT>/RESULTS.md

Deliverable: a clean PR-ready branch with commits, tests, updated docs, and a reviewable gpt bundle.

<environment_context>
  <cwd>/Users/mateobodon/Documents/Programming/Projects/kalshi-sys</cwd>
  <approval_policy>never</approval_policy>
  <sandbox_mode>danger-full-access</sandbox_mode>
  <network_access>enabled</network_access>
  <shell>zsh</shell>
</environment_context>
