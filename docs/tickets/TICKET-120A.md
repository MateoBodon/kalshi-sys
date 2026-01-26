# TICKET-120A

## Goal
Make NO-GO readiness markdowns actionable by adding a `How to fix` section to pilot readiness output and eliminate remaining absolute-path leaks (including ramp global reasons).

## Scope
- Update `src/kalshi_alpha/exec/pilot_readiness.py` to add a `How to fix` block on NO-GO.
- Harden `src/kalshi_alpha/exec/reports/ramp.py` so rendered global reasons cannot contain absolute paths.
- Expand/adjust tests to assert pilot readiness markdown contains `How to fix` when NO-GO and that readiness markdowns have no absolute paths.
- Do **not** change gate thresholds, pricing/strategy math, or broker behavior.

## Acceptance Criteria
- `pilot_readiness.render_markdown` includes `## How to fix` when overall decision is NO_GO or any global/series NO-GO exists.
- Ramp markdown still includes `How to fix` and global reasons cannot leak absolute paths.
- Pytest asserts absence of absolute paths in both readiness markdown variants.
- Tests assert `How to fix` appears in pilot readiness markdown for a NO-GO scenario.

## Plan
1. Inspect pilot readiness and ramp markdown renderers to identify NO-GO and reason handling (`src/kalshi_alpha/exec/pilot_readiness.py`, `src/kalshi_alpha/exec/reports/ramp.py`).
2. Add `How to fix` block to pilot readiness markdown when any NO-GO is present; sanitize global reasons before ramp markdown rendering.
3. Update tests to cover pilot readiness NO-GO `How to fix` and absolute-path checks (`tests/test_pilot_readiness.py`, `tests/test_scoreboard.py`).
4. Update ticket/run logs and repo docs (`docs/agent_runs/...`, `docs/PROGRESS.md`, `CHANGELOG.md`, and related docs if artifacts change).
5. Run required tests (`pytest -q tests/test_pilot_readiness.py tests/test_scoreboard.py` and `pytest -q`).

## Notes
- Staying on the existing branch per user request.
