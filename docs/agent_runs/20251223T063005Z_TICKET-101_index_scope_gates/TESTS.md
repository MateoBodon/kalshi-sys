# Tests & Runs

1) pytest -q
- Exit code: 0
- Result: 117 passed, 741 skipped

2) make monitors
- Exit code: 2
- Output: /bin/sh: python: command not found

3) PYTHON=python3 make monitors
- Exit code: 0
- Output summary:
  - ev_gap: NO_DATA
  - fill_vs_alpha: NO_DATA
  - ev_seq_guard: NO_DATA
  - freeze_window: ALERT — Freeze active for CPI
  - drawdown: OK
  - ws_disconnect_rate: OK
  - auth_error_streak: OK
  - kill_switch: OK

4) python3 -m kalshi_alpha.exec.preflight_index --offline
- Exit code: 1
- Output: PRECHECK index: NO-GO reasons=4 series=INX,NASDAQ100,INXU,NASDAQ100U

5) python3 -m kalshi_alpha.exec.supervisor_index --series INXU --dry-run --offline --max-runtime-seconds 5
- Exit code: 0
- Output: SUPERVISOR preflight: NO-GO reasons=4 series=INXU (broker=dry)
