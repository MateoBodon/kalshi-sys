# Results

Status: PASS

## Summary
- Verified systemd unit uses venv python, no PYTHONPATH, paper-only default, and StartLimit directives are in [Unit].
- Confirmed runbook references match unit name/path.
- Confirmed scipy/pandas are required by index runtime imports; EC2 `pip install -e .` succeeds without SciPy source builds.
- Captured EC2 systemd/journalctl evidence that service ran.

## Checklist Evidence (sanitized)

Systemd unit (no PYTHONPATH, StartLimit in [Unit], dry-run):
```
Environment=SUPERVISOR_BROKER=dry
ExecStart=/opt/kalshi-sys/.venv/bin/python -m kalshi_alpha.exec.supervisor_index --loop --sleep-seconds 20 --series INXU --dry-run --broker ${SUPERVISOR_BROKER}
StartLimitIntervalSec=300
StartLimitBurst=5
```

Runbook unit path:
```
/opt/kalshi-sys/deploy/systemd/supervisor_index.service -> /etc/systemd/system/kalshi-supervisor-index.service
```

EC2 proof (systemctl show):
```
ExecStart={ path=/opt/kalshi-sys/.venv/bin/python ; argv[]=/opt/kalshi-sys/.venv/bin/python -m kalshi_alpha.exec.supervisor_index --loop --sleep-seconds 20 --series INXU --dry-run --broker ${SUPERVISOR_BROKER} ; ... }
FragmentPath=/etc/systemd/system/kalshi-supervisor-index.service
User=kalshi
```

EC2 proof (systemctl status snippet):
```
Active: active (running)
... /opt/kalshi-sys/.venv/bin/python -m kalshi_alpha.exec.supervisor_index --loop --sleep-seconds 20 --series INXU --dry-run --broker dry
```

EC2 proof (journalctl snippet):
```
[REDACTED_HOST] python[183714]: SUPERVISOR preflight: GO reasons=0 series=INXU (broker=dry)
[REDACTED_HOST] python[183714]: [supervisor_index] 2025-12-22T03:58:10+00:00 running window hourly-1000 for series INXU
```

Dependency sanity:
- `preflight_index` imports `kalshi_alpha.strategies.index.model_polygon`, which imports `pandas` and `scipy.stats`.
- `hourly_above_below` imports `scipy.stats.skewnorm`.

EC2 pip install (no SciPy source build):
```
Requirement already satisfied: scipy>=1.11.0 ...
Requirement already satisfied: pandas>=2.2.0 ...
```

## Secrets hygiene
- No API keys/PEMs/tokens were logged or committed; logs contain only variable names (missing_env) and redacted host identifiers.

## Bundle
- docs/gpt_bundles/gpt_bundle_ticket-009_20251222_191125Z_TICKET-009_correctness_checklist.zip

## Follow-ups
- None.
