You are Codex running in Codex CLI inside the kalshi-sys repo.

WORK ONLY ON Ticket #4 (RETRY): “Pilot safety enforced at broker boundary (+ tests)”.
Hard scope: ONLY index ladder markets: INX / INXU / NASDAQ100 / NASDAQ100U.
STOP-THE-LINE: Do NOT enable LIVE trading by default. Do NOT relax pilot constraints. Do NOT touch secrets. Follow AGENTS.md.

You MUST satisfy BOTH:
(A) Ticket #4 technical acceptance criteria (broker-boundary safety + tests), AND
(B) Audit-trail acceptance criteria (diffs + run logs complete), because last attempt failed review due to missing diffs.

Workflow (required): explore → implement → test → document. No long upfront plan.

BRANCH + RUN LOG (required)
1) Create feature branch:
   git checkout -b codex/TICKET-004_pilot_broker_boundary_retry

2) Set:
   RUN_NAME_NEXT="$(date -u "+%Y%m%d_%H%M%SZ")_TICKET-004_pilot_broker_boundary_retry"

3) Create run log dir with REQUIRED files (must be filled, no “Pending”):
   docs/agent_runs/${RUN_NAME_NEXT}/
     - README.md (goal, summary, commands, tests, artifacts, risks)
     - prompt.md (this exact prompt)
     - commands.log (commands + key stdout/stderr; keep errors)
     - diff.patch (REAL git patch; full content, no "..." placeholders)
     - artifacts.json (paths + descriptions, non-empty)
     - META.json (run_name, ticket_id, branch, start/end UTC, network_access yes/no, web_search_used yes/no)
     - RESULTS.md (paths to artifacts + final gpt-bundle path)
     - TESTS.md (exact tests run + outcome)
     - external_facts.md ONLY if you use web search

Ticket #4 acceptance criteria (must satisfy)
1) Live mode cannot run without explicit acknowledgement and correct environment.
   - “Paper” is the default posture; live must be opt-in and fail-closed if ack/env missing.

2) Pilot mode enforces at broker boundary (fail-closed):
   - maker-only: reject crossing orders
   - orders outside window guard rejected
   - size > caps, or too many concurrent ladders rejected
   - kill switch blocks submits and prevents cancel/replace spam

3) Tests:
   - pytest -q passes
   - integration-ish test: simulate a crossing order and assert rejection
   - test(s): kill switch blocks submit AND does not loop cancel/replace
   - test(s): live-mode requires explicit ack gate (no ack => fail closed)

4) Docs:
   - update docs/PROGRESS.md and repo root PROGRESS.md if both exist
   - update CHANGELOG.md
   - gate stays PAPER

IMPORTANT PROCESS REQUIREMENTS (caused last FAIL)
- You MUST generate docs/agent_runs/${RUN_NAME_NEXT}/diff.patch and ensure it is NON-EMPTY:
    git diff --patch --no-color > docs/agent_runs/${RUN_NAME_NEXT}/diff.patch
  Then sanity-check:
    test -s docs/agent_runs/${RUN_NAME_NEXT}/diff.patch

- At the end you MUST create a review bundle and verify it contains a non-empty DIFF.patch:
    make gpt-bundle TICKET=TICKET-004_pilot_broker_boundary RUN_NAME=${RUN_NAME_NEXT}
    unzip -l <BUNDLE_PATH> | sed -n '1,200p'
  Record the bundle path in docs/agent_runs/${RUN_NAME_NEXT}/RESULTS.md.

(1) EXPLORE (fast; record key findings in README.md)
- Locate last-mile submit/cancel/replace path (the *actual* code path that would hit the network):
  - src/kalshi_alpha/brokers/kalshi/live.py (or equivalent live broker)
  - src/kalshi_alpha/core/execution/order_queue.py (or equivalent)
  - where proposals are converted into broker calls
- Locate pilot config loader + runtime flags:
  - configs/pilot.yaml (+ any index-only override)
  - src/kalshi_alpha/exec/pilot/config.py and runtime wiring
- Locate window guard logic:
  - src/kalshi_alpha/exec/window_guard.py
  - src/kalshi_alpha/sched/windows.py
- Locate kill switch sentinel path + how it’s checked.
- Locate/confirm TOB access path for maker-only check (in-memory snapshot, recent logger, or fetch).

(2) IMPLEMENT (fail-closed broker-boundary enforcer)
- Add a broker-boundary guard that runs immediately before ANY submit/cancel/replace that could touch the live API.
  - If pilot mode is enabled, enforce:
    - allowed series: index only (INX/INXU/NASDAQ100/NASDAQ100U).
      - If configs/pilot.yaml currently includes CPI or other macro series, tighten it OR add a dedicated index-only pilot config and ensure supervisor_index uses it.
    - max_contracts_per_order and max bins / max active ladders as defined by pilot config.
    - window guard: reject if outside allowed window or within freeze/cancel buffer.
      - If window cannot be determined, FAIL CLOSED with explicit reason.
    - maker-only:
      - Determine “crossing” using best bid/ask from a safe TOB source.
      - If TOB is unavailable or stale beyond a small threshold, FAIL CLOSED (reject) with explicit reason.
    - kill switch:
      - If kill switch is engaged, block submits.
      - Ensure order_queue does not spam cancel/replace; at most one cancel-all attempt per cycle (or similar deterministic bound).
- Live mode ack gate:
  - Require explicit acknowledgement flag (CLI or env) AND required environment/secrets present.
  - If missing: fail closed (raise/return error) BEFORE any network call.
  - Ensure default remains PAPER / dry-run (no live submits unless explicitly requested + acknowledged).

(3) TEST (offline, fixture-based)
- Add tests under tests/ (new files OK):
  - crossing_order_rejected():
      * create stub orderbook with bid/ask
      * create proposed order that would cross
      * assert broker-boundary guard rejects with a specific reason string/code
  - kill_switch_blocks_submit_and_no_spam():
      * create temp kill switch file
      * assert submit path returns blocked
      * assert cancel/replace loop bounded (no repeated calls)
  - live_requires_ack():
      * attempt to create/use live broker without ack => must fail closed
- Run:
    pytest -q
- Record exact output in docs/agent_runs/${RUN_NAME_NEXT}/TESTS.md.

(4) DOCUMENT
- Fill docs/agent_runs/${RUN_NAME_NEXT}/README.md with:
  - what guard is enforced where (file paths + function names)
  - what is still NOT enforced (known gaps)
  - risks/todos
- Update docs/PROGRESS.md + PROGRESS.md:
  - note Ticket #4 RETRY replaces the previous unreviewable run
  - include evidence links: test file paths + run dir
- Update CHANGELOG.md:
  - “Ticket #4: enforce pilot constraints at broker boundary; add safety tests”

(5) COMMITS (small, logical)
- Make small commits; each commit body must include:
  - Tests: <exact commands you ran>

(6) BUNDLE FOR REVIEW
- Generate review bundle and record the bundle path in RESULTS.md:
    make gpt-bundle TICKET=TICKET-004_pilot_broker_boundary RUN_NAME=${RUN_NAME_NEXT}
- Verify bundle contains non-empty DIFF.patch and run log diff.patch.

Now begin by exploring the repo (rg/rg --files), then implement.
