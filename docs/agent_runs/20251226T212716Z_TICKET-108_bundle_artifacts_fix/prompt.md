You are working in repo: kalshi-sys. Follow AGENTS.md as binding. This is a high-risk trading system: prefer correctness + auditability; PAPER must remain the default; do not enable live trading by default; do not leak secrets.

TICKET: TICKET-108
RUN_NAME: 20251226TICKET108_bundle_artifacts_fix  (use a proper UTC run name format like 20251226T<HHMMSS>Z_TICKET-108_bundle_artifacts_fix)

Goal: Fix gpt-bundle completeness so per-ticket bundles include generated calibration/readiness artifacts (fillcalib + pilot readiness + future calibration age reports). This ticket exists because TICKET-104’s bundle was missing artifacts listed in ARTIFACTS.md.

Work requirements (do not write a long upfront plan; just do the work in this order):
1) EXPLORE
   - Read AGENTS.md and docs/DOCS_AND_LOGGING_SYSTEM.md and docs/PLAN_OF_RECORD.md.
   - Locate how `make gpt-bundle` is implemented (Makefile target + underlying script).
   - Identify current allowlist/selection rules for what gets copied into per-ticket bundle and what goes into the .zip.
   - Reproduce the bug locally in a controlled way:
     - Create tiny dummy artifacts:
       - data/proc/fillcalib/curves_SMOKE.json
       - reports/fillcalib/SMOKE.md
       - reports/pilot_ready.json
       - reports/pilot_readiness.md
       - reports/calibration/SMOKE.md
     - Generate a bundle and verify via `unzip -l <bundle>.zip | grep -E "(fillcalib|pilot_ready|pilot_readiness|calibration)"`.
     - Confirm these are currently missing (baseline).

2) IMPLEMENT
   - Update the gpt-bundle builder to include, when present:
     - data/proc/fillcalib/*.json
     - reports/fillcalib/*.md
     - reports/pilot_ready.json and reports/pilot_readiness.md
     - reports/calibration/** (or at least reports/calibration/*.md)
   - Keep bundle size bounded: do NOT include entire reports/ wholesale; include only the specific subpaths above.
   - Add fail-closed logic:
     - If docs/agent_runs/<RUN_NAME>/ARTIFACTS.md lists any of the above artifacts (or generally any file path that exists in the workspace) and the bundler would omit it, the bundle build should fail with a clear error listing missing-from-bundle paths.
   - Update docs/DOCS_AND_LOGGING_SYSTEM.md to explicitly state fillcalib + pilot readiness + calibration age artifacts are included in per-ticket bundles (not just telemetry).

3) TEST
   - Run: `pytest -q` (required).
   - Add a regression test for the bundler selection logic:
     - The test should create a temp workspace structure (or use a test helper) with the dummy artifacts above and assert the resulting zip contains them (use `zipfile` to inspect members).
     - The test should also cover the fail-closed behavior when ARTIFACTS.md references an existing file that the bundler would omit.
   - Run the bundle build command for this ticket and show a short `unzip -l` excerpt in RESULTS.md (no secrets).

4) DOCUMENT
   - Create run log directory: docs/agent_runs/<RUN_NAME>/
     - Include: RUN.md, COMMANDS.md, TESTS.md, DIFF.patch, FILES_TOUCHED.md, ARTIFACTS.md, CITATIONS.md, NOTES.md, META.json.
     - IMPORTANT: the repo’s bundler appears to require docs/agent_runs/<RUN_NAME>/prompt.md; create/update that too (store this prompt there).
   - Update docs/PROGRESS.md: mark TICKET-104 as FAIL (missing artifacts in bundle) and add TICKET-108 as in-progress/done with what changed.
   - Update docs/CODEX_SPRINT_TICKETS.md: append TICKET-108 at bottom and set Status appropriately.
   - Update CHANGELOG.md with a dated entry describing the bundle completeness fix.

Git workflow constraints:
- Create a feature branch: codex/TICKET-108_bundle_artifacts_fix
- Make small logical commits (e.g., “bundle include paths”, “bundle verifier fail-closed”, “tests + docs”)
- Each commit body must include “Tests: …” with the exact commands you ran.

Finish:
- Generate the per-ticket bundle for review:
  PYTHON=python3 make gpt-bundle TICKET=TICKET-108 RUN_NAME=<RUN_NAME>
- Record the bundle path in docs/agent_runs/<RUN_NAME>/RESULTS.md
- Ensure the new bundle zip includes the dummy artifacts and (if they exist) any real fillcalib outputs.
