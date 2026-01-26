# RUN SUMMARY

Goal: Fix gpt-bundle completeness for calibration/readiness artifacts and enforce fail-closed checks.

Approach:
- Located the gpt-bundle target + verifier, reproduced bundle generation with dummy artifacts.
- Added a Python staging helper to centralize bundle selection and enforce ARTIFACTS.md inclusion checks.
- Updated Makefile to call the staging helper and include calibration subtrees.
- Added regression tests for bundler selection + fail-closed behavior.
- Regenerated the per-ticket bundle and captured a zip excerpt in RESULTS.

Key decisions:
- Bundle staging now copies `reports/calibration/**` to capture calibration-age reports in subdirectories.
- Fail-closed checks now occur during staging using ARTIFACTS.md before zipping.

Risks/notes:
- Baseline reproduction already included the dummy artifacts; missing entries could not be reproduced locally.
- ARTIFACTS.md omits the bundle zip to avoid self-bundle verification failures; bundle path is recorded in RESULTS.md.
