# Known Issues and Limitations

- **No recent fills in committed artifacts**: `reports/scoreboard_7d.md` and `reports/scoreboard_30d.md` report "No ledger data available," and `reports/pilot_readiness.md` shows insufficient data for all index series.
- **GO/NO-GO failures due to stale feeds**: `reports/_artifacts/go_no_go.json` lists stale Cleveland nowcast and Treasury yields, plus `heartbeat_stale` and `monitors_stale` reasons.
- **Model simplifications**: Polygon-only model uses sqrt(time) scaling with Student-t/normal assumptions and no explicit conditioning on realized volatility or microstructure features (noted in agent logs).
- **Fill model calibration gap**: Maker fill probability heuristic exists but is not calibrated on real TOB fills (explicit TODO in agent logs).
- **AWS wiring gap for index supervisor**: `supervisor_index` is implemented, but AWS/EventBridge wiring is noted as pending (agent log TODO).
- **Type-check coverage gaps**: `pyproject.toml` includes `mypy` ignore sections for several modules (scoreboard, CPI/claims/teny strategies, tests), which may mask typing regressions.
