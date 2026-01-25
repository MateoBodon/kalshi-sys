# COMMANDS.md
- `ls` (exit 0)
- `cat PROJECT.md` (exit 0)
- `cat AGENTS.md` (exit 0)
- `cat PROGRESS.md` (exit 0)
- `git status -sb` (exit 0)
- `cat docs/PROGRESS.md` (exit 0)
- `cat CHANGELOG.md` (exit 0)
- `date -u +%Y-%m-%d` (exit 0)
- `date -u +%Y%m%dT%H%M%SZ` (exit 0)
- `mkdir -p docs/agent_runs/20260125T023806Z_TICKET-111_project_state_refresh` (exit 0)
- `pytest -q` (exit 0)
- `cat <<'EOF' > docs/agent_runs/20260125T023806Z_TICKET-111_project_state_refresh/RUN.md` (exit 0)
- `cat <<'EOF' > docs/agent_runs/20260125T023806Z_TICKET-111_project_state_refresh/NOTES.md` (exit 0)
- `cat <<'EOF' > docs/agent_runs/20260125T023806Z_TICKET-111_project_state_refresh/TESTS.md` (exit 0)
- `cat <<'EOF' > docs/agent_runs/20260125T023806Z_TICKET-111_project_state_refresh/ARTIFACTS.md` (exit 0)
- `cat <<'EOF' > docs/agent_runs/20260125T023806Z_TICKET-111_project_state_refresh/CITATIONS.md` (exit 0)
- `{
  echo "# FILES_TOUCHED.md";
  {
    git diff --name-only;
    git ls-files --others --exclude-standard;
    echo "docs/agent_runs/20260125T023806Z_TICKET-111_project_state_refresh/DIFF.patch";
  } | sort -u | sed 's/^/- /';
} > docs/agent_runs/20260125T023806Z_TICKET-111_project_state_refresh/FILES_TOUCHED.md` (exit 0)
- `{
  echo "# FILES_TOUCHED.md";
  {
    git diff --name-only;
    git ls-files --others --exclude-standard;
    echo "docs/agent_runs/20260125T023806Z_TICKET-111_project_state_refresh/DIFF.patch";
  } | sort -u | sed 's/^/- /';
} > docs/agent_runs/20260125T023806Z_TICKET-111_project_state_refresh/FILES_TOUCHED.md` (exit 0)
- `git add -A` (exit 0)
- `git diff --cached > docs/agent_runs/20260125T023806Z_TICKET-111_project_state_refresh/DIFF.patch` (exit 0)
- `git add docs/agent_runs/20260125T023806Z_TICKET-111_project_state_refresh/DIFF.patch` (exit 1: ignored by .gitignore)
- `cat .gitignore` (exit 0)
- `git add -f docs/agent_runs/20260125T023806Z_TICKET-111_project_state_refresh` (exit 0)
- `git status -sb` (exit 0)
- `git reset HEAD docs/agent_runs/20260125T023806Z_TICKET-111_project_state_refresh` (exit 0)
- `git status -sb` (exit 0)
- `git commit -m "chore: snapshot repo state" -m "Tests: pytest -q"` (exit 0)
- `git status -sb` (exit 0)
