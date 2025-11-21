Session: Add historical Kalshi quote loader, wire real strikes into polygon backtest, and add maker fill model with tests/fixtures. Priority: SPX/NDX ladders only; keep maker-only and fee-aware, offline fixtures.

Plan:
- Inspect existing drivers/backtest wiring for index ladders.
- Build kalshi_index_history driver + fixtures/tests.
- Integrate real strikes/fill model into polygon backtest/CLI; extend tests and docs.

Session: Add index paper ledger + scoreboard for DRY runs (SPX/NDX only).

Plan:
- Add JSONL paper ledger writer for index dry trades and wire into dry broker.
- Build scoreboard CLI reading ledger, aggregating EV stats by day/series/window.
- Add tests, docs (AGENTS.md), and session log after smoke run.
