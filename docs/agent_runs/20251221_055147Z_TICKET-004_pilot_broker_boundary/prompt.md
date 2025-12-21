<environment_context>
You are Codex running in Codex CLI inside the kalshi-sys repo.
</environment_context>

Work ONLY on Ticket #4: “Pilot safety enforced at broker boundary (+ tests)”.
Hard scope: INX / INXU / NASDAQ100 / NASDAQ100U index ladders only.
STOP-THE-LINE: Do NOT enable LIVE trading by default. Do NOT relax pilot constraints. Do NOT touch secrets. Follow AGENTS.md.

Workflow (required): explore → implement → test → document. No long upfront plan.

BRANCH + RUN LOG (required)
- Create branch: codex/TICKET-004_pilot_broker_boundary
- Set RUN_NAME_NEXT = $(date -u "+%Y%m%d_%H%M%SZ")_TICKET-004_pilot_broker_boundary
- Create run log dir: docs/agent_runs/${RUN_NAME_NEXT}/ with:
  - README.md (goal, summary, commands, tests, artifacts, risks)
  - prompt.md (this exact prompt)
  - commands.log (commands + key stdout/stderr; keep errors)
  - diff.patch (REAL git patch; must include full content for all changed/new files, no "..." placeholders)
  - artifacts.json (paths + descriptions)
  - META.json (run_name, ticket_id, branch, start/end UTC, network_access yes/no)
  - RESULTS.md (paths to artifacts + gpt bundle path)
  - TESTS.md (exact tests run + outcome)
  - external_facts.md ONLY if you use web search

Ticket #4 acceptance criteria (must satisfy)
1) Live mode cannot run without explicit acknowledgement and correct environment.
2) Pilot mode enforces at broker boundary (fail-closed):
   - maker-only (reject crossing orders)
   - orders outside window guard rejected
   - size > caps or too many concurrent ladders rejected
   - kill switch blocks submits and prevents cancel/replace spam
3) Tests:
   - pytest -q passes
   - add an integration-ish test: simulate a crossing order -> must reject
   - add test(s) for kill switch submit-block and for live-mode explicit ack gate
4) Docs: update docs/PROGRESS.md + CHANGELOG.md; keep gate status PAPER (we are not live).

(1) EXPLORE (fast)
- Locate the last-mile submit path:
  - src/kalshi_alpha/brokers/kalshi/live.py
  - src/kalshi_alpha/core/execution/order_queue.py (or equivalent)
  - where execute_broker() routes proposals to broker
- Locate pilot config:
  - configs/pilot.yaml + loader (kalshi_alpha.exec.pilot.config)
- Locate kill switch sentinel logic and where cancel/replace happens.
- Identify how “maker-only” is currently enforced (if only at strategy layer, we must duplicate enforcement at submit boundary).

(2) IMPLEMENT (broker-boundary enforcement; fail-closed)
- Add a broker-boundary guard that runs immediately before any live submit / cancel / replace.
  - If pilot mode is enabled, enforce:
    - allowed series only (INX/INXU/NASDAQ100/NASDAQ100U)
    - max_contracts_per_order <= pilot cap
    - max unique bins <= pilot cap (or max active ladders; use whatever existing concept exists)
    - window guard: reject if now is past freeze/cancel buffer for the relevant window (use existing scheduler metadata; if missing, FAIL CLOSED with an explicit reason)
    - maker-only: reject if order would cross current TOB.
      - Preferred: if there is already an in-memory orderbook snapshot or recent TOB for the market at submit time, use it.
      - If not available, do a single lightweight orderbook fetch before submit (only in pilot/live path), then reject if crossing.
      - If you cannot obtain TOB safely/deterministically, FAIL CLOSED (reject) and log a reason.
    - kill switch: if kill switch sentinel exists, reject submits AND do not spam cancel/replace.
- Live mode hard gate:
  - Ensure that even if someone tries to call live broker code, it requires BOTH:
    - explicit CLI acknowledgement flag (e.g., --i-understand-the-risks) AND
    - required env/secrets present
  - If any requirement is missing: raise/return an error that prevents any network submit.

(3) TEST (must be offline/fixture-based; no secrets)
- Add/extend tests under tests/ that:
  - Crossing order rejection:
    - Provide a stub/fake orderbook with best bid/ask
    - Attempt to submit an order that crosses (e.g., buy at >= best ask / sell at <= best bid)
    - Assert broker-boundary enforcer rejects with a specific reason
  - Kill switch:
    - Create a temp kill switch file path
    - Assert submit path returns “blocked” and does NOT attempt cancel/replace loops
  - Live ack:
    - Construct live broker invocation without the ack flag and assert it fails closed
- Run: pytest -q
- Record exact output summary in docs/agent_runs/${RUN_NAME_NEXT}/TESTS.md

(4) DOCUMENT
- Update docs/PROGRESS.md with Ticket #4 entry (gate stays PAPER).
- Update CHANGELOG.md with a short Ticket #4 line.
- In README.md for the run log, explicitly list:
  - which enforcement checks are now guaranteed at broker boundary
  - any remaining known gaps

(5) BUNDLE FOR REVIEW
- Generate a new review bundle and record its path in RESULTS.md:
  make gpt-bundle TICKET=TICKET-004_pilot_broker_boundary RUN_NAME=${RUN_NAME_NEXT}

IMPORTANT: diff.patch must be a real patch. Use:
  git diff --patch --no-color > docs/agent_runs/${RUN_NAME_NEXT}/diff.patch
and verify it includes the new/modified test files and any new modules (no "..." placeholders).

Now begin by exploring the repo (rg/rg --files), then implement.
