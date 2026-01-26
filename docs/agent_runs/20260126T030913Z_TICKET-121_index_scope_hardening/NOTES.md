# NOTES

- `run_preflight` already passed a scope to freshness summaries, but accepted arbitrary `freshness_scope` values.
- The macro-stale preflight test lacked assertions and had trailing assertions mis-indented in the next test.
- Index supervisor runbook already described preflight; added explicit index-only freshness invariant.
