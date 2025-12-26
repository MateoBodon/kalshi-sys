You are Codex working in the kalshi-sys repo. Complete: Ticket #9 RETRY — EC2 verification of venv-based systemd unit (no PYTHONPATH hacks).

Hard constraints (binding):
- Follow AGENTS.md (stop-the-line rules; no secrets in logs; paper-only defaults).
- Do NOT make live trading possible by default. PAPER must remain default posture.
- Do NOT fake evidence. If you can’t access EC2, record “BLOCKED” with exact missing prereqs and stop.

Set identifiers:
- TICKET="TICKET-009_packaging_systemd_hardening_ec2_verify"
- RUN_NAME="$(date -u +"%Y%m%d_%H%M%SZ")_${TICKET}"
- BRANCH="codex/${TICKET}"

Create run log dir and required files up front:
- mkdir -p "docs/agent_runs/$RUN_NAME"
- Write these files (initial stubs ok, fill them as you go):
  - docs/agent_runs/$RUN_NAME/PROMPT.md  (paste this entire prompt)
  - docs/agent_runs/$RUN_NAME/COMMANDS.md (every command you run + key output snippets; redact hosts/keys)
  - docs/agent_runs/$RUN_NAME/RESULTS.md  (summary + acceptance evidence + bundle path)
  - docs/agent_runs/$RUN_NAME/TESTS.md    (exact test commands + results)
  - docs/agent_runs/$RUN_NAME/META.json   (run_name, ticket_id, branch, start/end UTC, environment=AWS/local, network_access, web_search_used)
  - docs/agent_runs/$RUN_NAME/diff.patch  (git diff or git show at end)
  - docs/agent_runs/$RUN_NAME/artifacts.json (paths to any artifacts)

Work in small commits; every commit body must include:
- Tests: <exact commands run>

0) Explore (no long plan):
- Read AGENTS.md and docs/CODEX_SPRINT_TICKETS.md (Ticket #9 acceptance).
- Read deploy/systemd/supervisor_index.service and docs/runbooks/aws_supervisor_index.md.
- Confirm the intended ExecStart uses /opt/kalshi-sys/.venv/bin/python and unit is paper-only (has --dry-run and defaults broker=dry).

1) Implement (only if needed):
- If you find any remaining PYTHONPATH references in the systemd unit or AWS runbook, remove them (fail-closed).
- If the unit depends on files/paths not documented (e.g., /etc/kalshi/kalshi-supervisor.env), make the runbook explicit.
- Keep changes minimal.

2) Local tests (always):
- Run: pytest -q
- Record in TESTS.md and COMMANDS.md.

3) EC2 verification (the actual acceptance):
Goal: On a fresh Ubuntu EC2 instance, start kalshi-supervisor-index.service WITHOUT manual PYTHONPATH tweaks, and capture proof.

Assumptions you may make (do NOT ask unless blocked):
- Operator has an EC2 host reachable via SSH and can sudo.
- Operator can provide required secrets via /etc/kalshi/kalshi-supervisor.env (DO NOT log contents).
- Repo can be cloned on the instance (private repo auth is handled out-of-band).

Do this via a reproducible command sequence (record everything in COMMANDS.md, but redact the host/IP and any tokens):
A) On EC2, as ubuntu (or appropriate user):
- Ensure system deps: python3.11-venv, git, build essentials if needed.
- Create /opt/kalshi-sys, clone repo there, checkout the branch you created.
- Create venv: /opt/kalshi-sys/.venv and pip install -U pip wheel
- pip install -e . (this must pull scipy/pandas wheels successfully on Ubuntu; if it fails, capture the error and fix appropriately—do not hack around with PYTHONPATH)
B) Install systemd unit:
- Copy deploy/systemd/supervisor_index.service to /etc/systemd/system/kalshi-supervisor-index.service
- systemctl daemon-reload
C) Prepare env file:
- Create /etc/kalshi/kalshi-supervisor.env with required vars (POLYGON_API_KEY, KALSHI keys, etc.).
- In logs, only show variable NAMES, not values.
D) Start service:
- systemctl enable --now kalshi-supervisor-index.service
E) Capture acceptance proof (sanitized):
- systemctl status kalshi-supervisor-index.service --no-pager
- systemctl cat kalshi-supervisor-index.service
- systemctl show kalshi-supervisor-index.service -p ExecStart -p FragmentPath -p User --no-pager
- journalctl -u kalshi-supervisor-index.service -n 200 --no-pager
Evidence must show:
- ExecStart points at /opt/kalshi-sys/.venv/bin/python
- No Environment=PYTHONPATH in the unit
- Service is running (or at least starts cleanly) with paper-only defaults (--dry-run)

If the service fails:
- Fail closed.
- Capture the error output (journalctl excerpt).
- Fix root cause properly (packaging, missing deps, wrong paths, permissions) and re-run the above until it passes.
- Do not “fix” by reintroducing PYTHONPATH hacks unless explicitly approved and documented as the chosen approach.

4) Documentation updates (required):
- Update PROGRESS.md:
  - Mark Ticket #9 as DONE only if the EC2 proof is captured in this run log.
  - Link to docs/agent_runs/$RUN_NAME/README.md (or RESULTS.md) and summarize what was proven.
- If any change affects ops/runbooks, update docs/runbooks/aws_supervisor_index.md accordingly.

5) Finish:
- Update docs/agent_runs/$RUN_NAME/RESULTS.md with:
  - PASS/FAIL
  - the exact acceptance proof commands + key outputs (sanitized)
  - any follow-ups needed (e.g., Ticket #10 CloudWatch validation)
- Write META.json end_utc, environment, network_access, etc.
- Generate bundle for review and record its path:
  make gpt-bundle TICKET="$TICKET" RUN_NAME="$RUN_NAME"

Stop-the-line:
- If you cannot obtain EC2 proof, do not claim completion. Mark FAIL, record why, and still generate the bundle.
