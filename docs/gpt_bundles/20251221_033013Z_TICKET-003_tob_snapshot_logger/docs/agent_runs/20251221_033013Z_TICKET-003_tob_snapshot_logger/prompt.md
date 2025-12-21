<environment_context>
You are Codex running in Codex CLI inside the kalshi-sys repo.
</environment_context>

Work ONLY on Ticket #3: “TOB snapshot logger + fill-calibration dataset skeleton”.
Hard scope: INX / INXU / NASDAQ100 / NASDAQ100U index ladders only (hourly + daily close windows).
Do NOT modify or enable LIVE trading. Do NOT relax pilot constraints. No secrets in logs.

Follow AGENTS.md exactly (STOP-THE-LINE rules apply).

REQUIREMENTS (must follow):
- Workflow: explore → implement → test → document. No long upfront plan.
- Use a feature branch: codex/TICKET-003_tob_snapshot_logger
- Small commits; each commit body must include: Tests: <exact commands>
- Create a run log dir:
  docs/agent_runs/<RUN_NAME_NEXT>/
  where RUN_NAME_NEXT = YYYYMMDD_HHMMSSZ_TICKET-003_tob_snapshot_logger

Run log must include (per AGENTS.md + reviewer loop):
- README.md (goal, summary, commands, tests, artifacts, risks)
- prompt.md (this exact prompt)
- commands.log (commands + key stdout/stderr; preserve errors)
- diff.patch (git diff or git show)
- artifacts.json (paths + descriptions)
- external_facts.md (ONLY if you use web search)
Additionally create:
- META.json (run_name, ticket_id, branch, start/end UTC, network_access yes/no)
- RESULTS.md (link to produced artifacts + gpt bundle path)
- TESTS.md (exact tests run + outcome summary)

Ticket #3 acceptance criteria (must satisfy):
1) For a dry-run supervisor session, TOB snapshots are written:
   - timestamped
   - series + window labeled
   - depth-limited and size-bounded
2) A dataset builder converts snapshots to a derived table ready for calibration:
   - includes mid, bid/ask, depth, our quote price/size (or quote-intent), time-to-expiry
3) Tests:
   - pytest -q passes
   - add fixture-based tests for schema + bounds (no live keys required)
4) Expected artifacts:
   - data/raw/kalshi/tob/*.jsonl (or under a per-run subdir)
   - data/proc/fillcalib/*.parquet
   - reports/fillcalib/README.md (schema + how to use)
   - run log + docs/PROGRESS.md + CHANGELOG.md updates

TASKS

(1) EXPLORE (fast, concrete)
- Find existing code for:
  - supervisor loop / CLI entrypoint: src/kalshi_alpha/exec/supervisor_index.py
  - any existing Kalshi WS/orderbook client: src/kalshi_alpha/brokers/kalshi/ws_client.py (or similar)
  - any existing telemetry/log sinks: src/kalshi_alpha/exec/telemetry/* or datastore helpers
  - any existing ledger/event logging for “dry-run orders” / “quote intents”
- Confirm whether `python -m kalshi_alpha.exec.supervisor_index --help` currently works.
  - If not, add a minimal CLI entrypoint (argparse) so Ticket #3 commands are runnable.
  - Do NOT change defaults to live; default must remain dry-run safe.

(2) IMPLEMENT — TOB snapshot logging (fail-closed, bounded)
- Add a `--record-tob` flag to supervisor_index (and plumb into the relevant runner) that:
  - when enabled, subscribes to orderbook/top-of-book for the relevant index ladder markets being scanned
  - writes JSONL snapshots to: data/raw/kalshi/tob/<RUN_NAME_NEXT>.jsonl (or data/raw/kalshi/tob/<RUN_NAME_NEXT>/tob.jsonl)
- Snapshot schema (minimum):
  - run_id (RUN_NAME_NEXT)
  - ts_utc (ISO)
  - series, window_label, window_ts_utc (ISO), window_ts_et (ISO)
  - market_ticker (and market_id if available)
  - bid_price, bid_size, ask_price, ask_size
  - optional: top_levels (depth-limited list of {price,size,side} up to N levels)
  - optional: quote_intent fields at decision time (price/size/side) if available
- Hard bounds:
  - depth limit: N <= 5 levels per side (prefer 1–3)
  - record size limit: truncate any snapshot that would exceed ~10KB; never write full books
  - do not log auth headers, tokens, or websocket URLs with credentials

(3) IMPLEMENT — quote-intent logging (minimal, dry-run friendly)
- In the scan/decision path (just before the broker submit, even in dry-run), log a “quote intent” event that includes:
  - ts_utc, run_id, market_ticker, series, window_label/window_ts
  - intended side, price, size
  - a reference to the most recent TOB snapshot timestamp (or include TOB fields inline)
- Goal: dataset builder can include “our quote price/size” without needing actual fills yet.

(4) IMPLEMENT — dataset builder tool
- Create: tools/build_fillcalib_dataset.py
  CLI:
  - --in <path to data/raw/kalshi/tob/> (directory or glob)
  - --out <path to data/proc/fillcalib/<name>.parquet>
- Output table columns (minimum):
  - run_id, ts_utc, series, window_label, market_ticker
  - bid, ask, mid, spread
  - bid_size, ask_size (and optional depth sums)
  - quote_price, quote_size, quote_side (from quote-intent; nullable if not present)
  - time_to_expiry_seconds (from window_ts_utc - ts_utc; clamp at >=0)
- Must be deterministic and not require network access.

(5) TEST (fixture-based)
- Add fixtures under tests/fixtures/fillcalib/:
  - a small synthetic tob.jsonl with a few snapshots (sanitized)
  - a small synthetic quote_intents.jsonl (if you log them separately)
- Add tests:
  - tests/tools/test_build_fillcalib_dataset.py: runs builder on fixtures and asserts schema + computed mid/spread/time_to_expiry
  - tests for snapshot bounds helper(s): ensures depth limit and size cap enforced
- Run: pytest -q (record in run log)

(6) DOCUMENT
- Create/update: reports/fillcalib/README.md
  - schema
  - how to run supervisor_index in dry-run with --record-tob
  - how to build dataset parquet
  - explicit note: this does NOT claim fills; it’s collecting inputs for later calibration
- Update docs/PROGRESS.md with Ticket #3 entry (gate status stays PAPER).
- Update CHANGELOG.md with a short entry.

(7) SMOKE (minimum, safe)
- Run a bounded dry-run that produces at least a few snapshots (avoid infinite runs):
  - If supervisor_index already supports a max-runtime/once flag, use it.
  - Otherwise add a small `--max-runtime-seconds` or `--max-snapshots` flag (default OFF) so the smoke run can exit cleanly.
- Then build a dataset parquet from the produced snapshots using tools/build_fillcalib_dataset.py.
- Record output paths in docs/agent_runs/<RUN_NAME_NEXT>/RESULTS.md

(8) BUNDLE FOR REVIEW
- Generate a new review bundle and record its path in RESULTS.md:
  make gpt-bundle TICKET=TICKET-003_tob_snapshot_logger RUN_NAME=<RUN_NAME_NEXT>

STOP CONDITIONS (do not proceed; request human review) if:
- You need to touch live broker auth/signing or change defaults in a way that could enable live trading.
- You cannot access TOB/orderbook data without changing security posture; in that case, implement only the logging interfaces + offline fixture path and document what endpoint/tool is needed.

Now begin by exploring the repo (use rg/rg --files), then implement.
