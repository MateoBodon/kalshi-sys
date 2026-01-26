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

- Date: 2026-01-26
- Decision: Sanitize absolute paths in readiness markdown by stripping root/drive prefixes (relative when possible, basename otherwise).
- Context: Pilot readiness outputs include calibration file paths that can be absolute on different machines.
- Options considered: Leave paths as-is; drop path columns entirely; make them relative/portable.
- Why: Preserve useful path context while ensuring portability across machines and bundles.
- Consequences: Readiness markdown no longer leaks machine-specific absolute paths.

- Date: 2026-01-26
- Decision: Sanitize ramp global reasons at markdown render time rather than mutating the policy JSON.
- Context: Global NO-GO reasons can include file paths and should be portable in markdown output.
- Options considered: Sanitize at policy generation; sanitize only when rendering markdown.
- Why: Keep JSON payloads intact for downstream tooling while ensuring human-readable outputs remain portable.
- Consequences: Ramp markdown avoids absolute paths without altering stored policy data.

- Date: 2026-01-26
- Decision: Enforce index-only freshness scope in `preflight_index` even if callers request a different scope.
- Context: `run_preflight` accepts a `freshness_scope` argument, but index runs must never be blocked by macro feed artifacts.
- Options considered: Honor caller-supplied scope; harden to `scope=index` for index preflight.
- Why: Prevent accidental macro gating while keeping index-only invariants explicit.
- Consequences: Non-index scope requests are ignored and the override is recorded in preflight details.

- Date: 2026-01-26
- Decision: Archive basis audit outputs by preserving project-relative paths under `--runlog/--archive-dir`.
- Context: All-series basis audits emit per-series JSON/MD artifacts with identical filenames across series.
- Options considered: Flat copy with series-prefixed filenames; nested copy mirroring project-relative paths.
- Why: Avoid filename collisions while keeping artifacts discoverable alongside other runlog evidence.
- Consequences: Runlogs now include `data/proc/basis/...` and `reports/basis/...` subtrees when archived.

- Date: 2026-01-26
- Decision: Use a temporary git stash in the GPT bundler to keep dirty-tree runs reproducible while keeping outputs in `artifacts/_local/gpt_bundles/`.
- Context: Bundles must be created even when the working tree is dirty, without writing tracked outputs.
- Options considered: Bundle directly from the dirty tree; use git archive from HEAD; stash and restore.
- Why: Stashing preserves the existing workflow while ensuring a clean, deterministic bundle snapshot and a reversible restore.
- Consequences: Bundling now performs a stash/apply/drop cycle by default; a `--no-stash` flag is required to opt out.
