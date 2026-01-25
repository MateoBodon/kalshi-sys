### Repo understanding

* **kalshi-sys / “Kalshi Alpha”** is a Python 3.11+ monorepo for **paper-first** research + scanning + guarded execution on **Kalshi ladder markets**, with **hard trading scope = index ladders only** (INX/INXU + NASDAQ100/NASDAQ100U).
* The core runtime loop is **ET-windowed**: resolve window → preflight/quality gates → ingest/fetch → model PMF → align to ladder strikes → compute EV-after-fees + fill/slippage → risk limits → broker (dry by default) → artifacts/reports.
* **Primary entrypoints** (from repo docs): `kalshi-scan` → `kalshi_alpha.exec.runners.scan_ladders:main`, plus `kalshi_alpha.exec.preflight_index` and `kalshi_alpha.exec.supervisor_index` for index windows.
* **Strategies/pricing** live under `src/kalshi_alpha/strategies/*` and `src/kalshi_alpha/core/pricing/*`, with **index PMF utilities** in `src/kalshi_alpha/models/pmf_index.py`.
* **Risk & safety** is layered: PAL/VaR/drawdown + pilot caps (`configs/pilot.yaml`) + **kill-switch file** (`data/proc/state/kill_switch`) + maker-only enforcement + explicit live acknowledgements.
* **Broker stack** includes a dry-run adapter and a live Kalshi adapter with **RSA‑PSS auth**, retries/rate limits, and audit logging; cancel/replace is coordinated via a FIFO order queue.
* **Freshness/quality gate configs** are explicit and index-scoped (`configs/freshness.index.yaml`, `configs/quality_gates.index.yaml`), with additional monitoring code under `src/kalshi_alpha/exec/monitors/*`.
* **Monitoring/reporting** produces window artifacts and rollups (scoreboards, pilot readiness, digests) intended to support a **fail-closed GO/NO‑GO** posture.
* **Tools** exist for evidence generation (notably **settlement basis audit** and **fill-calibration dataset building**) and for replay/housekeeping.
* The repo has a strong **test posture** (pytest suite + offline fixtures); canonical test command is `pytest -q`.
* The repo also has an **agent-run logging / bundle workflow** (`docs/agent_runs/*`, `tools/*bundle*`) and a generated **project_state/** “repo memory” snapshot intended for planning.
* **Important mismatch:** the included `project_state/` markdown snapshot is stamped **2025‑12‑22** (git SHA `a907a2e…`), while `_generated/git_head.txt` indicates current HEAD is **`31316e5…` on `main`** — i.e., the “repo memory” appears stale relative to current code.

---

### Current status

**What works:**

* The repo’s intended **index ladder pipeline** is clearly defined and safety-oriented: ET window scheduling, preflight, freshness/quality gates, risk limits, and dry-first execution flows are first-class.
* There is significant infrastructure for **evidence-based readiness**: basis audit tooling, fill calibration tooling, telemetry/monitor artifacts, pilot readiness/scoreboard concepts, and run logs.
* **Developer hygiene path** exists: formatting (ruff), lint/typecheck targets, and a large pytest suite with fixture-backed offline safety.

**What is missing:**

* A **current, regenerated `project_state/` snapshot** aligned to the present `main` HEAD (the provided one is older than HEAD).
* A single, unambiguous **“source of truth” for progress tracking** in the artifacts you provided (your root `PROGRESS.md` is basically empty, while the repo itself appears to use `docs/PROGRESS.md` per AGENTS).
* From the “readiness” perspective: the repo docs still surface **evidence gaps** (ledger/fill evidence, basis audit coverage, calibration-age visibility), but given the snapshot staleness, these need confirmation from current artifacts.

**What is broken:**

* **Working tree cleanliness / scaffold drift:** `_generated/git_status.txt` shows modified `.gitignore` + `AGENTS.md` and many untracked scaffold artifacts (e.g., backup `.bak.*` files and new docs/tools paths). That’s a real footgun for reproducibility and for agent-driven work.
* **Repo memory drift:** `project_state/*.md` claims it was generated from an older SHA/branch; relying on it for precise file/function-level decisions is risky until refreshed.

**Biggest risks (ranked):**

1. **Stale/incorrect “repo memory” leading to wrong edits** (project_state snapshot SHA ≠ current HEAD; planning could target moved/renamed behavior).
2. **Scope drift / accidental macro execution** (macro code exists; if family scoping defaults or flags are unclear, it can violate the non-negotiable index-only constraint).
3. **Accidental live enablement or degraded fail-closed posture** (high-risk domain; any ambiguity in broker arming, kill-switch, or gating must be treated as NO‑GO).
4. **Operational ambiguity** (which environment is authoritative, CloudWatch/logging status, durability of artifacts) — affects ability to build confidence without manual heroics.
5. **Docs/tooling duplication** (agentic scaffold wrappers vs existing bundle/build tools) creating confusion or subtly diverging behavior.

---

### Best next tickets (ranked)

#### 1) TICKET-111 — Refresh `project_state/` to match current `main` HEAD

* **Goal (1 sentence):** Regenerate and commit an up-to-date `project_state/` snapshot so the repo’s “AI memory” matches the current codebase.
* **Scope (what to change / not change):**

  * **Change:** Re-run the repo’s project_state build process on current `main`, update the generated markdown + `_generated/*` indices, and ensure metadata shows the correct SHA/branch/date.
  * **Do not change:** any production trading logic, risk logic, strategy math, or broker behavior.
* **Acceptance criteria (3–7 bullets):**

  * `project_state/INDEX.md` metadata matches current HEAD SHA (currently `_generated/git_head.txt` shows `31316e5…`).
  * `project_state/MODULE_SUMMARIES.md`, `FUNCTION_INDEX.md`, and `DEPENDENCY_GRAPH.md` reflect current code (no old branch references).
  * `project_state/KNOWN_ISSUES.md` and `OPEN_QUESTIONS.md` are reviewed/updated to align with current repo reality (keep short; no fiction).
  * Regeneration is reproducible via a documented command (in a run log).
  * `pytest -q` still passes.
* **Test command(s):**

  * `python tools/project_state_build.py`
  * `pytest -q`
* **Risk level:** low
* **Notes for Codex (pitfalls, files to touch):**

  * Touch: `tools/project_state_build.py` (if needed), `project_state/**`, and the run log under `docs/agent_runs/<RUN_NAME>/`.
  * Pitfall: don’t accidentally include large local caches or private data in generated inventories; keep ignores sane.

---

#### 2) TICKET-112 — Repo hygiene: reconcile the agentic scaffold with existing tooling and clean the working tree

* **Goal (1 sentence):** Get to a clean `git status`, remove/ignore scaffold backup debris, and ensure the “agentic hooks” call the repo’s canonical bundling/build tooling (no duplicated truth).
* **Scope (what to change / not change):**

  * **Change:** Remove or gitignore `.bak.*` and `.gitignore.append` style artifacts, decide whether `tools/agentic/*` are wrappers or should be deleted in favor of existing `tools/*bundle*`, and make AGENTS/runbooks consistent.
  * **Do not change:** strategy/pricing/risk logic; do not relax any live safeguards.
* **Acceptance criteria (3–7 bullets):**

  * `git status --porcelain` is empty after the ticket (or only contains intentionally tracked new files).
  * No backup artifacts (`*.bak.*`, `.gitignore.append`, etc.) are left as untracked clutter.
  * `AGENTS.md` “Agentic system hooks” point to the **actual** canonical commands/scripts used in this repo (wrappers must delegate cleanly).
  * Any newly introduced scaffold directories (`docs/_generated/`, `docs/_bundles/`, `docs/agent_runs/`, `project_state/_generated/`) are correctly gitignored/tracked as intended (explicit decision recorded).
  * `pytest -q` passes.
* **Test command(s):**

  * `git status --porcelain`
  * `pytest -q`
* **Risk level:** low
* **Notes for Codex (pitfalls, files to touch):**

  * Touch: `.gitignore`, `AGENTS.md`, and whichever wrapper scripts are meant to exist (`tools/agentic/*` vs existing `tools/gpt_bundle_builder.py`, `tools/verify_gpt_bundle.py`, `tools/project_state_build.py`).
  * Pitfall: don’t delete historically meaningful run logs or tracked ops proof docs; only remove true scaffold detritus.

---

#### 3) TICKET-113 — Add an automated “index-only” scope guard test suite (and fail closed on mis-scope)

* **Goal (1 sentence):** Make it mechanically hard to run macro flows accidentally by enforcing index-only defaults and adding regression tests around scope selection.
* **Scope (what to change / not change):**

  * **Change:** Add/extend tests that assert production entrypoints default to `FAMILY=index` and that macro execution requires an explicit opt-in signal; add minimal guardrails if a gap is found.
  * **Do not change:** macro model correctness, index pricing math, or execution sizing — this ticket is about **scope gating** only.
* **Acceptance criteria (3–7 bullets):**

  * New tests cover at least: `scan_ladders`, `preflight_index`, `supervisor_index`, and one pipeline entrypoint (`pipelines/daily` or `today`) for default family behavior.
  * Running a production entrypoint with no `--family` and no `FAMILY` env var results in **index-only** scope.
  * Attempting to run macro family without an explicit opt-in produces a **clear NO‑GO / error** (fail-closed) and does not place orders.
  * Tests run offline (no network, no secrets).
  * `pytest -q` passes.
* **Test command(s):**

  * `pytest -q`
* **Risk level:** medium (touches entrypoint behavior; must be careful not to break intended macro workflows for explicit users)
* **Notes for Codex (pitfalls, files to touch):**

  * Touch: likely `src/kalshi_alpha/exec/runners/scan_ladders.py`, `src/kalshi_alpha/exec/pipelines/*.py`, plus tests under `tests/exec/` or `tests/test_*`.
  * Pitfall: don’t block legitimate macro research runs when explicitly requested; the rule is “index by default, macro only by explicit opt-in.”

---

### If context is insufficient

* **Missing artifact:** a **fresh `project_state.zip` generated from current `main` HEAD** (your included `project_state/` markdown snapshot is stamped 2025‑12‑22 and references a different SHA/branch than `_generated/git_head.txt`).

  * **Generate with:** `python3 tools/agentic/project_state_refresh.py --zip` (or, if using the repo-native builder directly: `python tools/project_state_build.py`, then zip the resulting `project_state/` directory).
* **Missing artifact (optional but very helpful):** the latest `gpt_bundle.zip` for the most recent completed ticket/run (to verify “what’s broken” vs “already fixed” with concrete diffs + test outputs).

  * **Generate with:** `python3 tools/agentic/gpt_bundle.py --zip --ticket <TICKET_ID>` (or the repo-native tooling: `python tools/gpt_bundle_builder.py --ticket <TICKET_ID>` if that’s the canonical path).
