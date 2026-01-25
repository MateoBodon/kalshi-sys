# Project State Index

## Metadata
- Generated: 2026-01-10T11:43:04Z
- Git SHA: 31316e59451269689f2da173d8a9c6d9049d3d5e
- Branch: codex/TICKET-111_project_state_refresh
- Commands: `python tools/project_state_build.py`, `python3 tools/agentic/project_state_refresh.py --zip`, `rg --files`, `sed -n '1,200p' README.md`, `sed -n '1,200p' docs/PROGRESS.md`, `sed -n '1,200p' CHANGELOG.md`, `sed -n '1,200p' pyproject.toml`, `sed -n '1,200p' Makefile`

## How to read this folder
1) Start with `ARCHITECTURE.md` for the system overview.
2) Use `PIPELINE_FLOW.md` + `DATAFLOW.md` to understand runtime and data paths.
3) Review `MODULE_SUMMARIES.md` and `FUNCTION_INDEX.md` for code-level entrypoints.
4) Check `CURRENT_RESULTS.md`, `KNOWN_ISSUES.md`, and `OPEN_QUESTIONS.md` for status.
5) Consult `_generated/` for machine-readable indices.

## Navigation
- Architecture: `ARCHITECTURE.md`
- Module summaries: `MODULE_SUMMARIES.md`
- Function index: `FUNCTION_INDEX.md`
- Dependency graph: `DEPENDENCY_GRAPH.md`
- Pipeline flow: `PIPELINE_FLOW.md`
- Dataflow: `DATAFLOW.md`
- Experiments: `EXPERIMENTS.md`
- Current results: `CURRENT_RESULTS.md`
- Research notes: `RESEARCH_NOTES.md`
- Open questions: `OPEN_QUESTIONS.md`
- Known issues: `KNOWN_ISSUES.md`
- Roadmap: `ROADMAP.md`
- Config reference: `CONFIG_REFERENCE.md`
- Server environment: `SERVER_ENVIRONMENT.md`
- Test coverage: `TEST_COVERAGE.md`
- Style guide: `STYLE_GUIDE.md`
- Project_state changelog: `CHANGELOG.md`

## Generated artifacts
- `project_state/_generated/repo_inventory.json`
- `project_state/_generated/symbol_index.json`
- `project_state/_generated/import_graph.json`
- `project_state/_generated/make_targets.txt`
