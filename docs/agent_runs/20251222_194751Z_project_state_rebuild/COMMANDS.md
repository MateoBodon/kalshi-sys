# Commands

- git rev-parse HEAD
- git branch --show-current
- python --version
- uname -a
- rg --files
- sed -n '1,200p' README.md
- sed -n '1,200p' docs/PROGRESS.md
- sed -n '1,200p' CHANGELOG.md
- sed -n '1,200p' pyproject.toml
- sed -n '1,200p' Makefile
- python tools/project_state_build.py
- pytest -q
- zip -r docs/gpt_bundles/project_state_20251222_194751Z_a907a2e.zip docs/gpt_bundles/project_state_20251222_194751Z_a907a2e -x "**/__pycache__/**" -x "**/.pytest_cache/**"
