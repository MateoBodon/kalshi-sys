You are Codex running in the Codex CLI inside the kalshi-sys repo. Complete **TICKET-103** from docs/CODEX_SPRINT_TICKETS.md end-to-end.

HARD CONSTRAINTS (stop-the-line):
- Obey AGENTS.md as binding.
- Do NOT make live trading possible by default. PAPER/dry-run only.
- Do NOT add “fake fixes” (e.g., hardcoded outputs, disabling code paths) unless explicitly documented and required.
- Do NOT leak secrets into logs/docs/commits. Redact tokens/keys. Never print env var values.

WORKFLOW (do not write a long upfront plan; just do the work):
1) Create a feature branch: `codex/TICKET-103_telemetry_capture`.
2) Create a new run log folder: `docs/agent_runs/<RUN_NAME_NEXT>/` where:
   - `<RUN_NAME_NEXT>` = `$(date -u +%Y%m%dT%H%M%SZ)_TICKET-103_telemetry_capture`
   - Write these files as you go:
     - `PROMPT.md` (this exact prompt text)
     - `COMMANDS.md` (chronological commands + exit codes)
     - `RESULTS.md` (artifacts produced + final bundle path)
     - `TESTS.md` (tests run + outputs + exit codes)
     - `META.json` (run_name, ticket, start_utc, end_utc, git_sha_start)
     - Optional: `NOTES.md`, `FILES_TOUCHED.md`, `ARTIFACTS.md`, `CITATIONS.md` (only if web used)

IMPLEMENTATION GOAL (TICKET-103):
- During dry-run windows, collect:
  A) Top-of-book (TOB) snapshots (bounded depth/size)
  B) Quote intents (bounded size)
- Write compressed jsonl outputs:
  - `data/proc/telemetry/tob/<RUN_ID>.jsonl.gz`
  - `data/proc/telemetry/quote_intents/<RUN_ID>.jsonl.gz`
- Every telemetry row MUST include: `run_id`, `window_id`, `series`, `market_ticker`, `ts` (ISO8601).
- Add retention/housekeeping so telemetry cannot grow unbounded:
  - Document retention days + max bytes/window and implement enforcement.

EXPLORE (fast, concrete):
- Use `rg` to find existing telemetry/logging code and any existing `data/proc/telemetry` conventions.
- Inspect likely modules listed in the ticket:
  - `src/kalshi_alpha/exec/collectors/tob_logger.py`
  - `src/kalshi_alpha/exec/telemetry/sink.py`
  - `src/kalshi_alpha/exec/telemetry/shipper.py`
  - `src/kalshi_alpha/exec/supervisor_index.py`
  - `src/kalshi_alpha/exec/runners/micro_index.py`
  - `src/kalshi_alpha/exec/housekeep.py`
- Identify the minimal insertion points to emit:
  - TOB snapshots keyed by market + ts
  - Quote intents keyed by the proposal/run loop

IMPLEMENT (small commits):
- Add/extend a telemetry sink that can:
  - append jsonl entries
  - optionally gzip/rotate by run_id
  - enforce size limits (per file and/or per window_id)
- Ensure TOB capture is bounded (e.g., best bid/ask only, or depth N with N configurable).
- Ensure quote intents are bounded:
  - Only store the minimal data needed for later fill calibration (price, side, size, intended placement type, etc.)
  - Include correlation keys linking intent → market_ticker → window_id.
- Add housekeeping logic (retention) in `housekeep.py` or appropriate module; include documentation.

TEST (must be PAPER-safe):
- Run `pytest -q` and record results in `TESTS.md`.
- Add at least 1-2 unit tests that do NOT require network or secrets:
  - One that writes a few telemetry rows to a temp dir and verifies:
    - required fields exist
    - output is `.jsonl.gz`
    - size limiting/rotation works (at least minimal assertion)
  - One that exercises retention/housekeep on a temp tree.
- If you attempt any live dry-run command that requires secrets/network (e.g., `python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run`), STOP and ask for explicit human approval first. If not approved, document “BLOCKED (secrets/network)” in `TESTS.md` with what would have been run.

OPERATIONAL NOTE (new basis gate):
- If your dry-run path depends on preflight GO and is blocked by missing basis audits, you MAY generate the basis audit artifact locally first (PAPER-safe) using `tools/settlement_basis_audit.py` and store it in `data/proc/basis/...`.
- Do not relax/disable the basis gate to “make tests pass.” Instead, seed the required artifacts in tests using tmp paths.

DOCUMENT:
- Update `docs/DOCS_AND_LOGGING_SYSTEM.md` with the exact telemetry artifact paths + retention policy.
- Update `docs/PLAN_OF_RECORD.md` if telemetry artifacts/gating expectations change (even if gitignored, update the local file and ensure it appears in the bundle).
- Update `docs/PROGRESS.md` and `CHANGELOG.md` with a TICKET-103 entry and link to the run log.

COMMITS:
- Make small logical commits. In each commit body include:
  - “Tests: …” with the exact commands you ran (e.g., `pytest -q`).
- Do not commit large generated telemetry artifacts.

FINISH:
- Save a full diff to `DIFF.patch` (root) via `git diff > DIFF.patch`.
- Run: `make gpt-bundle TICKET=TICKET-103 RUN_NAME=<RUN_NAME_NEXT> PYTHON=python3`
- Record the bundle path in `docs/agent_runs/<RUN_NAME_NEXT>/RESULTS.md`.

Now start by creating the branch, run log folder/files, and exploring with `rg`.

---

Additional user instructions (2025-12-23)
- go ahead with this, 1. Patch tools/settlement_basis_audit.py to use the authenticated client and re-run the audit.
  2. Build data/proc/index_panel_polygon.parquet and run PYTHONPATH=src python3 -m jobs.calibrate_index_polygon_model so the index_polygon params exist.
  3. Rerun python -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run --record-tob. know thoughth that know its past 4pm so not sure if anything is active
