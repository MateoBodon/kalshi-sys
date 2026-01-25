# DECISIONS

Record non-obvious decisions. Keep it short.

Template:
- Date:
- Decision:
- Context:
- Options considered:
- Why:
- Consequences:

- Date: 2026-01-10
- Decision: Restore repo-specific Makefile and PLAN_OF_RECORD after running the bootstrap scaffold.
- Context: The scaffold overwrote existing ops and safety workflows that are part of this repo's runbooks.
- Options considered: Keep scaffold overwrites; merge original content back in.
- Why: Preserve established trading-system guidance while still installing the agentic toolchain.
- Consequences: Agentic scripts are available without losing existing workflows.
