You are Codex running in Codex CLI inside the kalshi-sys repo.

WORK ONLY ON Ticket #7: “Bundle / diff hygiene stop-the-line”.
Hard scope: do NOT change trading logic except where needed to enforce bundle/log correctness. Do NOT enable LIVE trading by default. Do NOT touch secrets. Follow AGENTS.md.

Goal:
- Make review bundles and run logs reliably complete and verifiable so Prompt-3 reviews stop failing on missing/empty diffs and missing run-log artifacts.

Acceptance criteria (must all be satisfied):
1) docs/agent_runs/<RUN_NAME>/diff.patch exists AND contains full patches (no placeholders / “content omitted”).
2) root DIFF.patch included in the review bundle is non-empty.
3) docs/agent_runs/<RUN_NAME>/README.md and RESULTS.md contain real content (no “Pending”, no empty stubs).
4) docs/agent_runs/<RUN_NAME>/META.json includes end_utc populated.
5) Run log protocol is enforced per AGENTS.md:
   - prompt.md (or PROMPT.md), commands.log, artifacts.json exist.
6) A bundle verification step exists and is documented + runnable (e.g., unzip -l + grep checks or a python verifier tool).
7) Add at least one automated test that fails if a bundle/run-log is missing required artifacts or contains placeholder diffs.

Workflow (required): explore → implement → test → document. No long upfront plan.

BRANCH + RUN LOG (required)
- Pick RUN_NAME_NEXT = YYYYMMDD_HHMMSSZ_TICKET-007_bundle_diff_hygiene (UTC).
- Create branch:
  git checkout -b codex/TICKET-007_bundle_diff_hygiene
- Create run log dir:
  mkdir -p docs/agent_runs/${RUN_NAME_NEXT}

In docs/agent_runs/${RUN_NAME_NEXT}/ write:
- prompt.md  (paste this prompt)
- commands.log (append every command you run + key outputs)
- TESTS.md (tests you ran + results)
- RESULTS.md (what changed + the bundle path you generated)
- README.md (goal + summary + links to artifacts)
- META.json (include run_name, ticket_id, branch, start_utc, end_utc, network_access, web_search_used)
- artifacts.json (list paths + short descriptions: diff.patch, bundle zip, etc.)
- diff.patch (git show or git diff output that includes full file bodies)

EXPLORE (do this first)
- Find how make gpt-bundle is implemented:
  rg -n "gpt-bundle|DIFF\\.patch|LAST_COMMIT|docs/gpt_bundles" -S .
  ls -la
  cat Makefile || true
  rg -n "zipfile|shutil\\.make_archive|subprocess\\.run\\(\\[\"zip\"" -S tools scripts src || true
- Inspect current docs requirements:
  sed -n '1,220p' AGENTS.md
  sed -n '1,260p' docs/DOCS_AND_LOGGING_SYSTEM.md
  sed -n '1,260p' docs/CODEX_SPRINT_TICKETS.md

IMPLEMENT
A) Add a bundle verifier tool (prefer Python):
- Create tools/verify_gpt_bundle.py (or similar) that:
  - opens a bundle zip
  - asserts root DIFF.patch exists and non-empty
  - asserts docs/agent_runs/<RUN_NAME>/diff.patch exists and contains real patch hunks (e.g., “diff --git”, “+++ b/”)
  - asserts required run-log files exist: README.md, RESULTS.md, META.json, prompt.md, commands.log, artifacts.json
  - rejects placeholder diff markers if any exist (e.g., lines containing “...”, “content omitted”, “(truncated)” in diff.patch)
  - prints a short PASS/FAIL report with explicit reasons and exits non-zero on failure

B) Wire verifier into the bundle generation path:
- Update Makefile / bundle script so:
  - make gpt-bundle ... produces the zip as before
  - then runs python tools/verify_gpt_bundle.py <zip_path>
  - and fails the make target if verification fails

C) Standardize run-log outputs (do NOT break existing runs):
- Update docs/DOCS_AND_LOGGING_SYSTEM.md and/or AGENTS.md only if needed to match reality.
- Ensure the bundler includes the run log folder with the required files.

TEST (required)
- Run fast suite minimum:
  pytest -q
  (record output in TESTS.md and commands.log)
- Add and run a new unit test for the verifier:
  - e.g., tests/test_gpt_bundle_verifier.py that creates a tiny temp zip with missing files and asserts verifier exits non-zero,
    and a “happy path” zip that passes.
  - Keep it fast, no network.

DOCUMENT (required)
- Update CHANGELOG.md with Ticket #7.
- Update PROGRESS.md with Ticket #7 entry (status: PAPER).
- In README.md + RESULTS.md in the run log, include:
  - the exact bundle path you generated
  - the verifier output summary

COMMITS (small, logical)
- Make small commits; each commit body must include:
  Tests: <exact commands you ran>

BUNDLE FOR REVIEW (required, last step)
- Generate bundle:
  make gpt-bundle TICKET=TICKET-007_bundle_diff_hygiene RUN_NAME=${RUN_NAME_NEXT}
- Record the resulting zip path in docs/agent_runs/${RUN_NAME_NEXT}/RESULTS.md

STOP-THE-LINE
- If you discover the bundler currently truncates diffs by design, do not “accept” truncation.
  Fix it so diffs include full file bodies, or explicitly include git show output in run-log diff.patch.

Now begin by exploring the repo with ripgrep and locating the gpt-bundle implementation.
