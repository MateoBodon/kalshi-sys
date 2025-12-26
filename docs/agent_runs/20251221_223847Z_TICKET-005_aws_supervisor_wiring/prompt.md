You are Codex running in Codex CLI inside the kalshi-sys repo.

HARD RULES (must follow):
- Read AGENTS.md first and follow it as binding.
- Do NOT enable live trading by default. PAPER/dry-run must remain the default posture.
- Do NOT add/print/seal any secrets (no API keys, no PEMs, no .env, no bearer tokens) in logs, docs, or commits.
- Do NOT commit local run logs or bundle zips (they are gitignored). Still create them locally and include them in the review bundle.
- Work in small commits. In EACH commit body include: `Tests: <exact commands you ran>`.

Ticket to complete: Ticket #5 — AWS / 24-7 supervisor wiring (audit-ready runbook + watchdog)
Goal: Produce an AWS-ready operational runbook and minimal wiring templates for 24/7 supervisor_index with alerts, plus break-glass procedures.

Process (do not write a long plan upfront):
1) Explore:
   - Open and read: docs/CODEX_SPRINT_TICKETS.md (Ticket #5), docs/PLAN_OF_RECORD.md, project_state/CONFIG_REFERENCE.md, project_state/KNOWN_ISSUES.md.
   - Locate the supervisor entrypoint and how it’s invoked:
     - Find `supervisor_index` module/CLI: rg -n "supervisor_index" -S src/ kalshi_alpha/ || rg -n "exec\.supervisor_index" -S .
   - Enumerate required env vars & configs (must be accurate):
     - rg -n "os\.getenv\(|environ\.get\(|POLYGON_API|KALSHI_" -S src/ || rg -n "POLYGON_API_KEY|KALSHI_API_KEY_ID|KALSHI_PRIVATE_KEY" -S .
     - Record the exact names and what uses them (for the runbook).
   - Identify existing heartbeat/monitor artifacts and paths:
     - rg -n "heartbeat|monitors|data_freshness" -S src/ configs/ reports/ project_state/
   - Identify any existing kill switch / cancel-all / broker-disable mechanisms:
     - rg -n "kill switch|KILL|cancel_all|cancel-all|BROKER|dry-run|paper" -S src/ configs/

2) Implement (docs + minimal templates; no new trading behavior):
   - Create new runbooks:
     - docs/runbooks/aws_supervisor_index.md
     - docs/runbooks/oncall_checks.md
   - Add ONE minimal deployment template (choose the simplest that fits existing repo structure; prefer systemd unless repo already uses ECS):
     - deploy/systemd/supervisor_index.service
       - Must include Restart policy, WorkingDirectory, User/Group placeholders, and a safe default ExecStart that uses --dry-run.
       - No hard-coded secrets. Use EnvironmentFile pointing to a local-only path (document using AWS SSM/Secrets Manager to populate it).
     - If systemd folder doesn’t exist, create it.
   - Optionally add configs/ops/ template if the repo already has ops config patterns:
     - configs/ops/supervisor_index_aws.example.yaml (only if consistent with current config loaders)

   Runbook requirements (must satisfy Ticket #5 acceptance criteria):
   - Deployment plan covers:
     - environment variables + secrets handling (AWS SSM Parameter Store / Secrets Manager)
     - restart policies (systemd or ECS)
     - log routing (journald -> CloudWatch agent OR ECS awslogs)
     - health checks: heartbeat + monitor staleness checks (reference exact artifact paths and thresholds from configs where possible)
     - alert conditions (what to alarm on, with concrete thresholds you can justify from existing configs)
   - Break-glass procedures included:
     - kill switch (how to stop trading immediately; prefer configuration/flag based, not code edits)
     - cancel-all (document how to cancel all orders safely; if no CLI exists, document manual steps + where to add a future tool)
     - disable live broker quickly (config/env var + restart)
   - Include a local smoke command in the runbook:
     - `python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run`
     - Also include a “no-keys” smoke (e.g., `--help`) if the dry-run still requires keys.

3) Test:
   - Run `pytest -q`.
   - Run a minimal CLI sanity check that does NOT require live keys:
     - `python -m kalshi_alpha.exec.supervisor_index --help` (or equivalent if module path differs)
   - Record exact commands + outcomes in docs/agent_runs/<RUN_NAME>/TESTS.md and commands.log.

4) Document + update progress:
   - Update docs/PROGRESS.md with a new entry for Ticket #5 (Gate status remains PAPER).
   - Update CHANGELOG.md with a concise Ticket #5 entry.
   - If the AWS wiring gap in project_state/KNOWN_ISSUES.md changes materially (e.g., runbook exists now), update that bullet to reflect reality.

5) Run log + bundle (REQUIRED):
   - Create a feature branch:
     - git checkout -b codex/TICKET-005_aws_supervisor_wiring
   - Create RUN_NAME dynamically:
     - RUN_NAME="$(date -u +"%Y%m%d_%H%M%SZ")_TICKET-005_aws_supervisor_wiring"
   - Create run log dir (local-only, gitignored):
     - mkdir -p "docs/agent_runs/$RUN_NAME"
   - Write these files:
     - docs/agent_runs/$RUN_NAME/prompt.md  (paste this prompt)
     - docs/agent_runs/$RUN_NAME/commands.log
     - docs/agent_runs/$RUN_NAME/TESTS.md
     - docs/agent_runs/$RUN_NAME/RESULTS.md
     - docs/agent_runs/$RUN_NAME/META.json   (include start_utc/end_utc, branch, web_search_used)
     - docs/agent_runs/$RUN_NAME/artifacts.json  (paths + descriptions)
     - docs/agent_runs/$RUN_NAME/diff.patch  (git diff or git show)
   - Ensure run log does NOT include secrets.
   - Finish by generating the review bundle and recording its path in RESULTS.md:
     - make gpt-bundle TICKET=TICKET-005_aws_supervisor_wiring RUN_NAME="$RUN_NAME"

Deliverable check before finishing:
- New runbooks exist with all required sections.
- At least one deploy template exists (systemd preferred).
- pytest passes; CLI help sanity check recorded.
- PROGRESS.md + CHANGELOG.md updated.
- Bundle generated successfully (and verifier passes automatically via Makefile).
