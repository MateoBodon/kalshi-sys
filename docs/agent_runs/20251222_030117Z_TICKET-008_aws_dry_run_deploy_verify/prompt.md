You are Codex running in the Codex CLI inside the kalshi-sys repo.

Ticket to complete: Ticket #8 — AWS dry-run deployment verification (EC2 + systemd + CloudWatch proof)

Hard constraints (must obey):
- Follow repo AGENTS.md (binding). If AGENTS.md conflicts with anything else, surface it and follow AGENTS.md.
- DO NOT make live trading possible by default. This ticket must run DRY / PAPER only.
- DO NOT leak secrets into logs, docs, diffs, or commits. Redact any AWS identifiers beyond what’s needed, and never print tokens/keys.
- Prefer correctness + auditability over new features.

Work style:
- Do not write a long upfront plan. Do: explore → implement → test → document.
- Use a feature branch. Small logical commits. Each commit message must include “Tests: …” with exact commands run.

Set variables:
- TICKET="TICKET-008_aws_dry_run_deploy_verify"
- RUN_NAME="<YYYYMMDD_HHMMSSZ>_TICKET-008_aws_dry_run_deploy_verify"  (use current UTC time)

Step 0 — Branch + run log scaffold
1) git checkout -b "codex/${TICKET}"
2) Create docs/agent_runs/${RUN_NAME}/ with the files required by AGENTS.md:
   - README.md (what this run is)
   - prompt.md (paste this entire prompt)
   - commands.log (append every command you run and key outputs)
   - RESULTS.md (start a “Status: IN PROGRESS” section)
   - TESTS.md (start empty; fill after tests)
   - META.json (include: ticket, run_name, start_utc now, end_utc null for now, network_access true/false, web_search_used true/false)
   - artifacts.json (start empty; fill with any produced artifacts/paths)
3) From this point: every command you run gets appended to commands.log with any critical stdout snippets.

Explore (repo inspection)
4) Locate existing AWS/runbook/systemd wiring for the index supervisor:
   - Find files mentioning: "kalshi-supervisor-index", "systemd", "CloudWatch", "journalctl", "heartbeat.json"
   - Identify the intended working directory, venv strategy, and whether the service uses `PYTHONPATH=src` or an installed package.
5) Confirm the service runs supervisor_index in DRY mode and cannot place live orders by default.

Implement (make AWS dry-run actually runnable)
6) Ensure there is a single, canonical systemd service definition for the dry-run index supervisor, and that it:
   - Runs: `python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run` (and any required args)
   - Emits logs including:
     - window selection
     - the single-line GO/NO-GO preflight summary (from Ticket #6)
     - heartbeat updates
   - Writes/updates `data/proc/state/heartbeat.json` and freshness monitor artifacts (whatever the repo defines as canonical)
   - Solves the src-layout import issue:
     - Either install the package on EC2 (preferred) OR set `Environment=PYTHONPATH=/path/to/repo/src` in the unit.
   - Uses a dedicated non-root user if the runbook already establishes one.
7) Add or update a runbook doc (in docs/ or project_state/) that is explicit and copy/pasteable:
   - EC2 setup steps (clone, venv, install, configs)
   - systemd enable/start/status commands
   - how to view logs locally (journalctl)
   - how CloudWatch logs are shipped and where to look (log group/stream naming)
   - how to stop/kill-switch the service safely
   - redaction policy (what MUST NOT be pasted into logs/docs)
8) If CloudWatch shipping config is missing or vague, add the minimal config needed (or a clearly-scoped “manual step” section) so a human can set it up deterministically.

Test (local + minimal real validation where possible)
9) Run the repo fast test suite minimum:
   - `pytest -q`
   Record this in docs/agent_runs/${RUN_NAME}/TESTS.md with the output summary.
10) Run a local smoke of supervisor_index in DRY mode to ensure:
   - It starts
   - It prints the single-line preflight summary
   - It writes/updates heartbeat.json
   Use explicit `PYTHONPATH=src` if needed, and record commands + key output in commands.log.

AWS verification (must attempt; if blocked, document exactly what’s missing)
11) Attempt the AWS dry-run verification required by the ticket:
   - Run kalshi-supervisor-index.service on an EC2 instance in dry mode for >= 2 windows.
   - Capture sanitized CloudWatch log excerpts showing:
     - window selection
     - GO/NO-GO summary line
     - heartbeat updates
   - Confirm `data/proc/state/heartbeat.json` updates and freshness monitor artifact updates.
   Record the proof snippets (sanitized) in docs/agent_runs/${RUN_NAME}/RESULTS.md.
   If you cannot perform AWS steps due to missing credentials/SSH/instance info, do NOT fake it:
     - Mark RESULTS.md as BLOCKED
     - Provide a minimal, exact command checklist for a human to execute to produce the required evidence
     - Still finish all code/doc changes that make the runbook executable.

Document + project state updates
12) Update:
   - docs/PROGRESS.md (add Ticket #8 entry; include Gate status PAPER; link this run log)
   - CHANGELOG.md (Ticket #8 entry)
   - project_state/CURRENT_RESULTS.md and/or project_state/KNOWN_ISSUES.md if the AWS verification changes the status (e.g., AWS verification no longer pending)

Finish
13) Set end_utc in META.json.
14) Ensure no secrets in diffs/logs: run a quick ripgrep scan for common key patterns and redact if needed.
15) Create the review bundle:
   - `make gpt-bundle TICKET=${TICKET} RUN_NAME="${RUN_NAME}"`
16) In docs/agent_runs/${RUN_NAME}/RESULTS.md, record the path to the generated zip bundle and summarize what changed + what proof was collected.

Commit rules
- Make small commits (e.g., “systemd unit fixes”, “runbook updates”, “docs/progress updates”).
- Each commit message must include a “Tests: …” line with exact commands actually run.
