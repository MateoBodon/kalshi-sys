TICKET-101 — Decouple index-only GO/NO-GO from macro freshness + add explicit scope in artifacts

You are working in the repo root. Follow these constraints strictly:
- Scope: ONLY index ladders (INX/INXU/NASDAQ100/NASDAQ100U). Do not touch macro pipelines except to ensure they do NOT interfere with index-only runs.
- Safety: do NOT enable any live trading. All runs must be dry-run / read-only.
- Workflow: explore → implement → test → document. Do not write a long upfront plan.

0) Setup hygiene
- Create a feature branch: codex/TICKET-101_index_scope_gates
- Create RUN_NAME from UTC now: "$(date -u +%Y%m%dT%H%M%SZ)_TICKET-101_index_scope_gates"
- Create run log dir: docs/agent_runs/$RUN_NAME/
- Start docs/agent_runs/$RUN_NAME/RUN.md with: goal, initial repo state (git sha), and a checklist of acceptance criteria.

1) Explore (fast, concrete)
- Locate and read the current index preflight + scan flow:
  - src/kalshi_alpha/exec/preflight_index.py
  - src/kalshi_alpha/exec/runners/scan_ladders.py
  - src/kalshi_alpha/exec/monitors/freshness.py
  - src/kalshi_alpha/core/gates/quality_gates.py
- Find where GO/NO-GO decisions are computed and where the go/no-go artifact is written.
- Identify exactly how “macro freshness” is currently entering the decision for index runs.
- Write a short exploration note in docs/agent_runs/$RUN_NAME/NOTES.md including:
  - current behavior (what blocks index runs today)
  - which config files are loaded in index runs (freshness + quality gates)
  - the smallest code/config change to scope freshness checks

2) Implement (small commits)
Implement these changes:
A) Decouple macro freshness from index scope:
- Ensure index-only entrypoints load index-specific freshness config:
  - configs/freshness.index.yaml (create if missing)
  - configs/quality_gates.index.yaml (create/adjust if missing)
- In freshness monitor / quality gates, add an explicit “scope” concept (string or enum) and a scoped filter so only index-scoped feeds are considered for index runs.

B) Add explicit scope field to go/no-go artifacts:
- The go/no-go artifact must include:
  - scope: "index"
  - scoped_blockers: [ ... ]  # only blockers relevant to that scope
  - unscoped_blockers (optional) must be empty for index runs or clearly labeled "ignored"
- Make sure the artifact is emitted by both:
  - python -m kalshi_alpha.exec.preflight_index
  - python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run (if it writes artifacts)

C) Prevent regressions:
- Add a test that simulates “macro stale” while index feeds are fresh and expects index preflight GO.
  - If the codebase already has fixtures or a way to inject feed freshness, use that.
  - If not, add a small deterministic fixture-based mechanism (no network) to set freshness states.

Commit rules:
- Make small commits (2–4). Each commit message should be descriptive.
- In each commit body, include a "Tests:" line listing what you ran.

3) Test (must run)
- Run: pytest -q
- Run: make monitors (if available; otherwise document why not)
- Run: python -m kalshi_alpha.exec.preflight_index (should be PAPER-safe)
- If it does not require secrets: run python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run for a short cycle.
Record all outputs/exit codes in docs/agent_runs/$RUN_NAME/TESTS.md.

4) Document (required)
- Update docs/PLAN_OF_RECORD.md:
  - Add a short note that index-only runs are scope-isolated from macro freshness.
  - Document which configs control index freshness gating.
- Update docs/PROGRESS.md:
  - Mark TICKET-101 completed and note any follow-ups.
- Update CHANGELOG.md with a short entry.

5) Run log completeness
In docs/agent_runs/$RUN_NAME/ add:
- FILES_TOUCHED.md (list)
- DIFF.patch (git diff saved at end)
- COMMANDS.md (commands executed)
- ARTIFACTS.md (any artifacts produced during tests)

Stop-the-line:
- If you discover that index runs share a global freshness registry that can’t be scoped without a larger refactor, implement the smallest safe scoping shim and clearly document the debt in NOTES.md (do not “half-fix” silently).
