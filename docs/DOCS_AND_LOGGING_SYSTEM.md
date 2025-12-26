# DOCS AND LOGGING SYSTEM — Traceability for kalshi-sys

Last updated: 2025-12-26

This repo is a high-risk trading system. The primary goal of this document is **auditability**:
- every run is traceable,
- every decision is reproducible,
- no “invisible state,”
- no secrets in logs.

---

## 1) Principles (non-negotiable)

1) **No invisible state**
- If it matters for trading, it must be recorded as an artifact (config, inputs, outputs, gate reasons).

2) **Run IDs everywhere**
- Every paper/pilot/live run must have a `run_id` and `window_id`.
- All telemetry rows must include: `run_id`, `window_id`, `series`, `market_ticker`, and an event timestamp.

3) **Fail-closed**
- If an artifact can’t be written (disk full / permissions / S3 down), treat it as NO-GO.

4) **Redaction by default**
- Assume logs are exfiltration risk. Never write secrets.

---

## 2) Directory layout (canonical)

### 2.1 Prompts
Store all human + agent prompts in:
- `docs/prompts/`

Naming convention:
- `docs/prompts/YYYYMMDD_TICKET-###_<short>.md`

Required header in each prompt file:
- date (UTC)
- ticket id
- git sha (if known)
- what the prompt is intended to do
- any external sources used (with retrieval date)

### 2.2 Agent runs
Every Codex/agent run MUST write a run log directory:
- `docs/agent_runs/<RUN_NAME>/`

RUN_NAME format:
- `<YYYYMMDDTHHMMSSZ>_TICKET-###_<short_slug>`

Example:
- `docs/agent_runs/20251223T021530Z_TICKET-101_index_scope_gates/`

Required files inside each run directory:
- `RUN.md` (human-readable summary; includes: goal, approach, key decisions, known risks)
- `COMMANDS.md` (commands executed + outputs/exit codes; paste excerpts not full secrets)
- `TESTS.md` (tests run; must include `pytest -q` or justified exception)
- `DIFF.patch` (or `git diff` saved at end of run)
- `FILES_TOUCHED.md` (bullet list of files modified)
- `ARTIFACTS.md` (paths produced + sha256 if feasible)
- `CITATIONS.md` (if any external facts were consulted; include retrieval date + URL)
- `NOTES.md` (open issues, follow-ups)

Optional but recommended:
- `CONFIG_SNAPSHOT/` (copies of relevant `configs/*.yaml` used in the run)
- `SCREENSHOTS/` (only if useful; no secrets)

### 2.3 Progress / changelog
These files are the “single pane of glass” and must be updated per ticket:
- `docs/PROGRESS.md` — current gate status + what changed + next blockers
- `CHANGELOG.md` — dated entries, one per ticket

### 2.4 Bundles (shareable snapshots)
We maintain two bundle types:

A) **Full project state bundles** (for audits)
- Path:
  - `docs/gpt_bundles/project_state_<YYYYMMDD>_<HHMMSSZ>_<gitsha7>/`
- Contents MUST include:
  - `project_state/*.md` (ARCHITECTURE, PIPELINE_FLOW, CURRENT_RESULTS, etc.)
  - `docs/` key plans (PLAN_OF_RECORD, DOCS_AND_LOGGING_SYSTEM, CODEX_SPRINT_TICKETS, PROGRESS)
  - `_generated/` indices (repo_inventory, symbol_index, function_index)
- Should also include a zip:
  - `docs/gpt_bundles/project_state_<...>.zip`

B) **Per-ticket bundles** (small, fast)
- Path:
  - `docs/gpt_bundles/ticket_<TICKET-###>_<YYYYMMDDTHHMMSSZ>_<gitsha7>/`
- Contents MUST include:
  - `docs/agent_runs/<RUN_NAME>/`
  - files changed in the ticket
  - any new reports/fixtures
  - telemetry artifacts when generated (`data/proc/telemetry/*` and `reports/ops/telemetry_volume_*.md`)
- Should also include a zip:
  - `docs/gpt_bundles/ticket_<...>.zip`

### 2.5 When to regenerate a full project_state bundle
Regenerate full project_state when any of these occur:
- new pipeline stage introduced or reordered,
- new live/pilot safety gate introduced,
- fee model changes,
- fill model calibration logic changes,
- ops/deploy wiring changes (systemd/CloudWatch),
- gate status changes (PAPER → PILOT, PILOT → LIVE).

Otherwise, per-ticket bundles are sufficient.

---

## 3) Naming conventions for reports and artifacts

### 3.1 Reports (human-readable)
- `reports/` is for Markdown summaries that a reviewer can read quickly.
- Organize by category:
  - `reports/paper/<YYYY-MM-DD>/...`
  - `reports/pilot/<YYYY-MM-DD>/...`
  - `reports/live/<YYYY-MM-DD>/...`
  - `reports/basis/<SERIES>/<YYYY-MM-DD>.md`
  - `reports/fillcalib/<ASOF_DATE>.md`
  - `reports/calibration/<ASOF_DATE>.md`
  - `reports/ops/<ASOF_DATE>.md`
  - `reports/ops/telemetry_volume_<YYYY-MM-DD>.md`
  - `reports/fees/<ASOF_DATE>.md`

### 3.2 Machine-readable artifacts
- `data/proc/` is the canonical store for machine-readable artifacts.
Suggested subdirs:
- `data/proc/state/` (kill switch, heartbeats, supervisor status)
- `data/proc/runs/<RUN_ID>/` (go/no-go, proposals, monitor events)
- `data/proc/telemetry/`
  - `tob/<RUN_ID>.jsonl.gz` (bounded TOB snapshots)
  - `quote_intents/<RUN_ID>.jsonl.gz` (bounded quote intents)
  - `ws_status/` (ws heartbeat + staleness summaries)
  - `runs/<RUN_ID>.json` (run metadata: preflight status, bounds, paths)
- `data/proc/basis/` (basis audits)
- `data/proc/basis/<SERIES>/<YYYY-MM-DD>.json` (daily basis summary: quantiles, per-window deltas, flip-risk flag)
- `data/proc/calibration/` (calibration outputs)
- `data/proc/fillcalib/` (fill curves)
- `data/proc/fillcalib/dataset_<ASOF_DATE>.parquet` (optional fillcalib dataset if small)

### 3.3 Market status guardrails
- The Polygon indices websocket collector checks `/v1/marketstatus/now` before REST fallback.
- If indices groups are closed/extended-hours, it suppresses fallback and logs `market_status` plus `serverTime` to avoid false stale alarms.
- Freshness monitor also consults `/v1/marketstatus/now` so closed/extended hours don’t mark `polygon_index.websocket` as stale.
- Ops CLI: `python -m kalshi_alpha.exec.market_status` (use `--json` for raw payload).

---

## 4) Redaction policy (secrets + PII)

### 4.1 Never log or commit
- API keys, tokens, secrets, cookies, session tokens
- Full request/response bodies that may contain auth headers
- Any personally identifying info

### 4.2 Allowed
- Environment variable NAMES only (e.g., `KALSHI_API_KEY`), never values
- Last-4 characters of IDs if needed for debugging (e.g., `...a1b2`)
- Hashes (sha256) of config files

### 4.3 Required sanitization
- Telemetry sinks must implement:
  - max depth / max bytes,
  - field allowlist,
  - truncation on large strings,
  - explicit redaction patterns for `*_KEY`, `*_TOKEN`, `Authorization`.

---

## 5) Retention policy (avoid ops foot-guns)

Minimum expectations:
- Telemetry must be bounded (size and count).
- Old artifacts should be compressed and/or pruned.
- Disk usage must be monitored and can trigger NO-GO.

Documentation requirement:
- Every ticket that adds a new telemetry stream must specify:
  - maximum bytes per window,
  - retention days,
  - pruning mechanism,
  - how it’s monitored.

Telemetry retention (index ladders):
- Max bytes per window (per stream): 256KB (bounded at write time).
- Per-record caps: TOB snapshots 10KB, quote intents 2KB.
- Retention days: 30 (pruned via `python -m kalshi_alpha.exec.housekeep --keep-days 30`).

---

## 6) Per-ticket checklist (must be satisfied before merge)

- [ ] Feature branch created (no direct commits to main)
- [ ] `pytest -q` run (or explicitly justified exception)
- [ ] Run log directory created: `docs/agent_runs/<RUN_NAME>/`
- [ ] `docs/PROGRESS.md` updated
- [ ] `CHANGELOG.md` updated
- [ ] Ticket marked updated in `docs/CODEX_SPRINT_TICKETS.md`
- [ ] If external facts were used: `CITATIONS.md` updated with retrieval date + URLs
- [ ] If artifacts produced: listed in `ARTIFACTS.md` and stored under `reports/` or `data/proc/`
