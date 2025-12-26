then i want you to go through this checklist and make sure everything is completed, after commit all changes, merge to main and push to origin, only make the final bundle after all of this is done so it is up to date, Ticket #9 correctness


Confirm deploy/systemd/supervisor_index.service has no PYTHONPATH and uses .venv/bin/python.


Confirm it is paper-only by default (--dry-run, broker defaults to dry).


Confirm StartLimit directives are in the correct systemd section ([Unit]), and the unit name/path matches the runbook.




Dependency sanity


Re-check whether adding scipy/pandas is truly necessary for the index ladder runtime path (it may be, but it’s heavy).


If you run on EC2: ensure pip install -e . succeeds without compiling SciPy from source.




Evidence quality


Do not merge “it should work” claims: require systemctl show -p ExecStart + journalctl snippets proving the service actually ran on EC2.




Secrets hygiene


Ensure no .env contents, private keys, Kalshi tokens, or API keys appear in COMMANDS.md, RESULTS.md, commit messages, or patches.
