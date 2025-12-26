<environment_context>
  <cwd>/Users/mateobodon/Documents/Programming/Projects/kalshi-sys</cwd>
  <approval_policy>never</approval_policy>
  <sandbox_mode>danger-full-access</sandbox_mode>
  <network_access>enabled</network_access>
  <shell>zsh</shell>
</environment_context>

You are Codex running in the Codex CLI inside the kalshi-sys repo.

Ticket to complete: Ticket #9 — Packaging/systemd import-path hardening

Hard constraints (must obey):
- Follow AGENTS.md (binding). If AGENTS.md conflicts with other docs, surface it and follow AGENTS.md.
- DO NOT enable or modify LIVE trading behavior. Paper/dry must remain default.
- DO NOT leak secrets into logs, docs, diffs, or commits.

Work style:
- Do not write a long upfront plan. Do: explore → implement → test → document.
- Use a feature branch. Small logical commits. Each commit message must include “Tests: …” with exact commands run.

Set variables:
- TICKET="TICKET-009_packaging_systemd_hardening"
- RUN_NAME="<YYYYMMDD_HHMMSSZ>_TICKET-009_packaging_systemd_hardening" (UTC)

Step 0 — Branch + run log scaffold (per AGENTS.md)
1) git checkout -b "codex/${TICKET}"
2) Create docs/agent_runs/${RUN_NAME}/ with REQUIRED files:
   - README.md, RESULTS.md, META.json, prompt.md, commands.log, diff.patch, artifacts.json
   - external_facts.md ONLY if you use web search
3) From now on: append every command + key outputs to commands.log.

Explore
4) Identify why we still rely on `PYTHONPATH=src`:
   - Inspect pyproject.toml / packaging config
   - Verify whether `pip install -e .` installs `kalshi_alpha` (src-layout) correctly
5) Inspect systemd units + runbooks:
   - deploy/systemd/supervisor_index.service (and any kalshi-supervisor-index.service templates)
   - docs/runbooks/* (AWS / oncall checks)
   - Confirm current unit directives are correct (StartLimit*, WorkingDirectory, User, restart policy)
6) Confirm defaults remain paper-only (dry broker default; no live by default).

Implement (pick ONE approach and make it deterministic)
7) Preferred approach: “editable install + venv python”:
   - Ensure a fresh venv can install the repo with dependencies:
     - `python -m venv .venv && source .venv/bin/activate`
     - `pip install -U pip`
     - `pip install -r requirements.txt` (or repo’s canonical requirements file)
     - `pip install -e .`
   - Update systemd unit(s) so they do NOT require `PYTHONPATH=src`:
     - ExecStart must call the venv python (absolute path) OR a stable wrapper script that activates venv and execs python.
     - Set WorkingDirectory to repo root.
     - Put StartLimitIntervalSec/StartLimitBurst in the correct systemd section (or remove if not needed).
     - Keep `--dry-run` in the unit by default and document how to keep it that way.
   - Add a small bootstrap script for EC2 (copy/pasteable) that:
     - clones repo
     - creates venv
     - installs deps + editable install
     - installs systemd unit
     - starts/stops/status checks
     - includes a “no secrets in logs” reminder

Test (local)
8) Run repo fast test suite minimum and record in TESTS.md:
   - `pytest -q`
9) Packaging smoke test that approximates “fresh EC2”:
   - Create a TEMP venv (e.g., /tmp/kalshi_pkg_smoke_venv)
   - Install deps + `pip install -e .`
   - Run WITHOUT PYTHONPATH:
     - `python -m kalshi_alpha.exec.preflight_index --offline --now "2025-12-22T10:50:00-05:00"`
     - `python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run --offline --now "2025-12-22T10:50:00-05:00" --no-ws-listen`
   - Record stdout + artifact paths in commands.log and artifacts.json
   - If this fails due to missing deps (e.g., scipy), fix the bootstrap to install the correct requirement set (do not hack around imports).

Document
10) Update:
   - docs/PROGRESS.md (Ticket #9 entry + evidence path)
   - CHANGELOG.md (Ticket #9 entry)
   - project_state/KNOWN_ISSUES.md: update AWS verification wording to reflect Ticket #8 proof, and clarify what’s still missing (e.g., “fresh EC2 bootstrap + always-on supervisor”).
11) Secret-scan before committing:
   - `git diff --cached | rg -n "API_KEY|PRIVATE_KEY|Bearer|BEGIN RSA|webhook" || true`

Commit discipline
12) Make small logical commits. Each commit must include “Tests: …” with exact commands run.

Finish
13) Set end_utc in META.json.
14) Generate review bundle and record its path in RESULTS.md:
   - `make gpt-bundle TICKET=${TICKET} RUN_NAME="${RUN_NAME}"`
