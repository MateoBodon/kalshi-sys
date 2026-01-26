# Agent runs

Tracked agent run logs (small files only). Use: `YYYYMMDDTHHMMSSZ_TICKET-####_slug/`.
Required files: `RUN.md`, `NOTES.md`, `COMMANDS.md`, `TESTS.md`, `DIFF.patch`,
`FILES_TOUCHED.md`, `ARTIFACTS.md` (and `CITATIONS.md` if external sources were used).
`RUN.md` must include a short manifest (commit hash, commands run, key inputs, UTC timestamp).
Bundling writes to `artifacts/_local/gpt_bundles/` (ignored by design) and is allowed on dirty trees; the tool stashes temporarily unless `--no-stash` is set.
Large dumps belong in `reports/_runs/` or `artifacts/_local/`.
