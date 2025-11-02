# agents.md

## Purpose
Define automation agents operating on this repo: roles, guardrails, credentials flow, and acceptance criteria.

## Agents

### Repo Automation Engineer (Codex)
- **Scope:** Code changes only. Lives under `src/kalshi_alpha/`, tests in `tests/`, docs in `docs/`.
- **Powers:** Edit code, add tests, update docs, open PRs. No secret rotation, no production deploys.
- **Guardrails:**
  - Conventional Commits only; no force-push.
  - Must run ruff/mypy/pytest before committing.
  - No plaintext secrets; use environment variables; update `.env.example`.
  - For live code: add/extend tests or mark with xfail + TODO explaining why.
- **Definition of Done (per change):**
  - Tests updated/passing; CI green.
  - Docs updated (RUNBOOK / CHANGELOG).
  - If touching risk limits, include rationale and updated guardrail tests.

### Ops Runner (Human)
- **Scope:** Scheduling, live toggles, credentials, monitoring, deployments.
- **Runbooks:** See `docs/RUNBOOK.md` (live smoke, pilot ramping, failure modes).
- **Kill switches:** Files monitored by the broker; setting them to on should halt submissions.

## Environments
- **Local/dev:** Offline by default; use fixtures and cassettes.
- **Live/pilot:** Minimal size, maker-only, limited markets.

## Credentials
- Env-var based. Example file: `.env.example`. For real secrets use $SECRET_MANAGER (Doppler/1Password/Vault).
