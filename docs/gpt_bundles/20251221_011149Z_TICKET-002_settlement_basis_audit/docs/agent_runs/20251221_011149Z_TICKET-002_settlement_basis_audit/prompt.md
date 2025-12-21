<environment_context>
  <cwd>/Users/mateobodon/Documents/Programming/Projects/kalshi-sys</cwd>
  <approval_policy>never</approval_policy>
  <sandbox_mode>danger-full-access</sandbox_mode>
  <network_access>enabled</network_access>
  <shell>zsh</shell>
</environment_context>

You are Codex running in Codex CLI inside the kalshi-sys repo.

Work ONLY on Ticket #2: “Settlement basis audit (Polygon vs Kalshi expiration value)”.
Hard scope: INX / INXU / NASDAQ100 / NASDAQ100U index ladder windows only (hourly + daily close).
Do NOT modify or enable LIVE trading. Do NOT relax pilot constraints. No secrets in logs.

Follow AGENTS.md exactly (STOP-THE-LINE rules apply).

REQUIREMENTS (must follow):
- Workflow: explore → implement → test → document. No long upfront plan.
- Use a feature branch: codex/TICKET-002_settlement_basis_audit
- Small commits; each commit body must include: Tests: <exact commands>
- Create a run log dir:
  docs/agent_runs/<RUN_NAME_NEXT>/
  where RUN_NAME_NEXT = YYYYMMDD_HHMMSSZ_TICKET-002_settlement_basis_audit

Run log must include (per AGENTS.md + for reviewer compatibility):
- README.md (goal, summary, commands, tests, artifacts, risks)
- prompt.md (this exact prompt)
- commands.log (commands + key stdout/stderr; preserve errors)
- diff.patch (git diff or git show)
- artifacts.json (paths + descriptions)
- external_facts.md (ONLY if you use web search)
Additionally create:
- META.json (basic metadata: run_name, ticket_id, branch, start/end UTC, network_access yes/no)
- RESULTS.md (link to produced reports + dataset paths + gpt bundle path)
- TESTS.md (exact tests run + outcome summary)

Ticket #2 acceptance criteria (must satisfy):
1) Implement a command that produces a daily report for any date + series:
   - basis distribution (mean/median/p95/p99)
   - “flip risk” flags where basis magnitude could change outcome near strikes
2) Reproducible:
   - saves raw inputs (Kalshi expiration/settlement value + Polygon values + timestamps)
   - can re-run from saved inputs without hidden live-only dependencies
3) Add at least one unit test with saved fixtures.
4) Produce expected artifacts:
   - reports/settlement_basis/<day>_<series>.md
   - data/proc/settlement_basis/<day>_<series>.parquet (or jsonl)

TASKS

(1) EXPLORE (fast, concrete)
- Locate:
  - Market discovery / window→ticker mapping for index ladders: likely src/kalshi_alpha/markets/discovery.py (confirm).
  - Any existing “expiration value / settlement value” fetch in Kalshi client wrappers.
  - Existing Polygon index fetch helpers (minute bars / snapshots).
  - Existing strike/bin parsing utilities used by scanners (reuse; do not re-invent parsing).
- Identify the minimal reliable fields you can use to compute:
  - kalshi_expiration_value (or equivalent settlement underlying value)
  - polygon_value_at_window (closest-at-or-before window timestamp)
  - nearest_strike_margin for flip-risk

(2) IMPLEMENT
- Create: tools/settlement_basis_audit.py
  CLI args:
  - --day YYYY-MM-DD (required)
  - --series {INX,INXU,NASDAQ100,NASDAQ100U} (required)
  - --out-report (optional; default reports/settlement_basis/<day>_<series>.md)
  - --out-data (optional; default data/proc/settlement_basis/<day>_<series>.parquet)
  - --use-cache (optional): if output dataset exists, recompute report from it without any API calls
  - --offline-fixtures (optional): run using local fixture JSON instead of network calls (must be used by tests)

- Output dataset schema (minimum columns; add more if cheap):
  - day (date), series (str)
  - window_ts_et (ISO), window_ts_utc (ISO)
  - kalshi_value (float) + kalshi_source_field (str) + kalshi_market_or_event_id (str)
  - polygon_value (float) + polygon_source (str) + polygon_ts_utc (ISO)
  - basis = polygon_value - kalshi_value
  - nearest_strike (float|None), nearest_strike_margin = abs(polygon_value - nearest_strike) (float|None)
  - flip_risk (bool): True iff nearest_strike_margin is not None AND abs(basis) >= nearest_strike_margin

- Report content (markdown):
  - Summary stats: count, mean, median, p95, p99 of basis and abs(basis)
  - Top-N windows by abs(basis)
  - Flip-risk table: windows where flip_risk=True with polygon_value, kalshi_value, basis, nearest_strike, margin
  - Explicit note: “Polygon is not settlement truth; Kalshi expiration value is the reference for settlement.”
  - Exact command used + git SHA (if available) for reproducibility

- Reproducibility requirement:
  - Always write the dataset first (raw inputs included in columns).
  - Report generation must be able to run from the dataset alone (no API calls) via --use-cache.

(3) TEST
- Add fixtures under tests/fixtures/settlement_basis/:
  - minimal Kalshi response fixture (containing settlement/expiration value fields you use)
  - minimal Polygon response fixture (containing window timestamp/value)
  - minimal market discovery fixture (or stub discovery mapping) with at least 2 windows and strikes

- Add at least one unit test:
  - tests/tools/test_settlement_basis_audit.py (or similar)
  - It must run in offline-fixtures mode and assert:
    - dataset written with expected columns
    - report written and includes p95/p99 lines
    - flip_risk True for at least one known case

- Run tests:
  - pytest -q
  - If pytest fails due to packaging, fix the repo-local developer path (prefer editable install guidance or test harness fix), but do not hack imports.

(4) DOCUMENT
- Create/update docs/prompts/TICKET-002_settlement_basis_audit.md (store a copy of this prompt + the exact commands you intend to run).
- Update docs/PROGRESS.md with a Ticket #2 entry (gate status still PAPER).
- Update CHANGELOG.md with a short entry.

(5) SMOKE (minimum)
- Run the tool at least twice (offline fixtures is OK for the agent run):
  - python tools/settlement_basis_audit.py --day 2025-11-10 --series INXU --offline-fixtures
  - python tools/settlement_basis_audit.py --day 2025-11-10 --series NASDAQ100U --offline-fixtures
- Record outputs/paths in docs/agent_runs/<RUN_NAME_NEXT>/RESULTS.md

(6) BUNDLE FOR REVIEW
- Inspect Makefile to confirm the correct invocation of gpt-bundle.
- Then generate a new review bundle and record its path in RESULTS.md:
  make gpt-bundle TICKET=TICKET-002_settlement_basis_audit RUN_NAME=<RUN_NAME_NEXT>

STOP CONDITIONS (do not proceed; request human review) if:
- You need to touch live broker auth, signing, or defaults.
- You find that “expiration value” is not available in the Kalshi APIs you can access; in that case, write a minimal stub + doc that clearly states what field is missing and what endpoint/field must be confirmed.

Now begin by exploring the repo (fast grep/open files), then implement.
