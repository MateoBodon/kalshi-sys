# Config Reference

## Key configuration files (semantics)

### configs/index_ops.yaml
- **Purpose**: index operations window definitions and scanner defaults.
- **Loader**: `kalshi_alpha.config.index_ops.load_index_ops_config`.
- **Key options**:
  - `window_hourly`: window definition for INXU/NASDAQ100U (supports `start_offset_min`, `end_at_target`, `cancel_buffer_seconds`).
  - `window_close`: window definition for INX/NASDAQ100 (supports `start`, `end`, `cancel_buffer_seconds`).
  - `min_ev_usd`: default minimum EV per order (code fallback: 0.05).
  - `max_bins_per_series`: default per-series bin cap (code fallback: 2).

### configs/fees.json and configs/fees/series.json
- **Purpose**: fee coefficients and rounding metadata for EV calculations.
- **Loaders**: `kalshi_alpha.exec.fees`, `kalshi_alpha.core.fees.index_series`.
- **Key options**:
  - `series.<SERIES>.maker_fee`, `series.<SERIES>.taker_fee`, `series.<SERIES>.coefficient`.
  - `rounding.mode`, `rounding.quantum`.

### configs/quality_gates.yaml / configs/quality_gates.index.yaml
- **Purpose**: GO/NO-GO thresholds for model metrics and data freshness.
- **Loader**: `kalshi_alpha.core.gates.quality_gates` and `kalshi_alpha.exec.gate_utils`.
- **Key options**:
  - `metrics.default` and `metrics.series.*` (CRPS/Brier thresholds).
  - `data_freshness` list with `namespace`, `timestamp_field`, `max_age_hours`/`max_age_seconds`.
  - `monitors` thresholds (e.g., `tz_not_et`, `non_monotone_ladders`).

### configs/pal_policy.yaml
- **Purpose**: PAL (Position and Loss) limits per series/strike.
- **Loader**: `kalshi_alpha.exec.scanners.index_scan_common` and `kalshi_alpha.exec.runners.scan_ladders`.
- **Key options**:
  - `series_policies.<SERIES>.default_max_loss`
  - `series_policies.<SERIES>.per_strike` (pattern -> max loss).

### configs/index_correlation.yaml
- **Purpose**: correlation-aware VaR caps for SPX/NDX families.
- **Loader**: `kalshi_alpha.risk.correlation`.
- **Key options**: `portfolio_limit`, `confidence_z`, `families.*.limit`, `families.*.tilt_limit`, `correlations.*`.

### configs/index_var.yaml
- **Purpose**: per-family VaR limits.
- **Loader**: `kalshi_alpha.risk.var_index`.
- **Key options**: `limits.SPX`, `limits.NDX`, `limits.OTHER`.

### configs/pilot.yaml
- **Purpose**: pilot session constraints (maker-only, max bins/contracts, allowed series).
- **Loader**: `kalshi_alpha.exec.pilot.config` and pilot runners.
- **Key options**: `pilot.allowed_series`, `pilot.max_contracts_per_order`, `pilot.max_unique_bins`, `pilot.require_live_broker`.

### configs/size_ladder.yaml
- **Purpose**: staged sizing caps by series.
- **Loader**: `kalshi_alpha.config.size_ladder`.
- **Key options**: `current_stage`, `stages.<stage>.per_series.<SERIES>.max_contracts/max_bins`.

### configs/freshness.yaml / configs/freshness.index.yaml
- **Purpose**: data freshness thresholds for monitors.
- **Loader**: `kalshi_alpha.exec.monitors.freshness`.
- **Key options**: `feeds.*` with `age_days`/`age_seconds`, `required`, `namespace`.

### configs/portfolio.yaml
- **Purpose**: factor vols and strategy betas used in risk preview.
- **Loader**: `kalshi_alpha.exec.runners.risk_preview`.

### configs/series.yaml
- **Purpose**: metadata about macro settlement sources and lag days.
- **Used by**: reporting and strategy metadata helpers.

### configs/fee_rules_watch.yaml
- **Purpose**: official Kalshi fee/rulebook URLs to monitor for changes.
- **Loader**: `monitor/fee_rules_watch.py`.

### configs/systemd/* and configs/logrotate/*
- **Purpose**: deployment templates for timers/services and log rotation.

---

## Full key index (auto-extracted)

## configs/fee_rules_watch.yaml

Keys (nested):
- targets
- targets.id
- targets.url

Referenced in code/docs:
- fee_rules_watch.yaml

## configs/fees/series.json

Keys (nested):
- extends
- notes

Referenced in code/docs:
- configs/fees/series.json
- series.json

## configs/fees.json

Keys (nested):
- rounding
- rounding.mode
- rounding.quantum
- series
- series.INX
- series.INX.coefficient
- series.INX.label
- series.INX.maker_fee
- series.INX.taker_fee
- series.INXU
- series.INXU.coefficient
- series.INXU.label
- series.INXU.maker_fee
- series.INXU.taker_fee
- series.NASDAQ100
- series.NASDAQ100.coefficient
- series.NASDAQ100.label
- series.NASDAQ100.maker_fee
- series.NASDAQ100.taker_fee
- series.NASDAQ100U
- series.NASDAQ100U.coefficient
- series.NASDAQ100U.label
- series.NASDAQ100U.maker_fee
- series.NASDAQ100U.taker_fee
- updated

Referenced in code/docs:
- configs/fees.json
- fees.json

## configs/freshness.index.yaml

Keys (nested):
- feeds
- feeds.polygon_index.websocket
- feeds.polygon_index.websocket.age_seconds
- feeds.polygon_index.websocket.label
- feeds.polygon_index.websocket.namespace
- feeds.polygon_index.websocket.required
- required_order

Referenced in code/docs:
- No direct filename/path reference found in scanned source files.

## configs/freshness.yaml

Keys (nested):
- feeds
- feeds.aaa_gas.daily
- feeds.aaa_gas.daily.age_days
- feeds.aaa_gas.daily.price_max
- feeds.aaa_gas.daily.price_min
- feeds.aaa_gas.daily.required
- feeds.bls_cpi.latest_release
- feeds.bls_cpi.latest_release.age_days
- feeds.bls_cpi.latest_release.required
- feeds.cleveland_nowcast.monthly
- feeds.cleveland_nowcast.monthly.age_days
- feeds.cleveland_nowcast.monthly.required
- feeds.dol_claims.latest_report
- feeds.dol_claims.latest_report.age_days
- feeds.dol_claims.latest_report.required
- feeds.nws_daily_climate
- feeds.nws_daily_climate.active_stations
- feeds.nws_daily_climate.age_days
- feeds.nws_daily_climate.required
- feeds.polygon_index.websocket
- feeds.polygon_index.websocket.age_seconds
- feeds.polygon_index.websocket.label
- feeds.polygon_index.websocket.namespace
- feeds.polygon_index.websocket.required
- feeds.treasury_10y.daily
- feeds.treasury_10y.daily.age_business_days
- feeds.treasury_10y.daily.expected_maturity
- feeds.treasury_10y.daily.required
- required_order

Referenced in code/docs:
- configs/freshness.yaml
- freshness.yaml

## configs/index_correlation.yaml

Keys (nested):
- confidence_z
- correlations
- correlations.NDX
- correlations.NDX.NDX
- correlations.NDX.SPX
- correlations.SPX
- correlations.SPX.NDX
- correlations.SPX.SPX
- families
- families.NDX
- families.NDX.limit
- families.NDX.series
- families.NDX.tilt_limit
- families.SPX
- families.SPX.limit
- families.SPX.series
- families.SPX.tilt_limit
- portfolio_limit

Referenced in code/docs:
- configs/index_correlation.yaml
- index_correlation.yaml

## configs/index_ops.yaml

Keys (nested):
- max_bins_per_series
- min_ev_usd
- window_close
- window_close.cancel_buffer_seconds
- window_close.end
- window_close.start
- window_hourly
- window_hourly.cancel_buffer_seconds
- window_hourly.end_at_target
- window_hourly.start_offset_min

Referenced in code/docs:
- index_ops.yaml

## configs/index_var.yaml

Keys (nested):
- limits
- limits.NDX
- limits.OTHER
- limits.SPX

Referenced in code/docs:
- configs/index_var.yaml
- index_var.yaml

## configs/logrotate/kalshi-alpha

Keys (nested):
- Unable to parse or no structured keys detected.

Referenced in code/docs:
- kalshi-alpha

## configs/pal_policy.example.yaml

Keys (nested):
- default_max_loss
- per_strike
- per_strike.CPI-2025-10
- per_strike.CPI-2025-11
- per_strike.CPI-2025-12
- series

Referenced in code/docs:
- configs/pal_policy.example.yaml
- pal_policy.example.yaml

## configs/pal_policy.yaml

Keys (nested):
- series_policies
- series_policies.INX
- series_policies.INX.default_max_loss
- series_policies.INX.per_strike
- series_policies.INX.per_strike.KXINX-*
- series_policies.INXU
- series_policies.INXU.default_max_loss
- series_policies.INXU.per_strike
- series_policies.INXU.per_strike.KXINXU-*
- series_policies.NASDAQ100
- series_policies.NASDAQ100.default_max_loss
- series_policies.NASDAQ100.per_strike
- series_policies.NASDAQ100.per_strike.KXNASDAQ100-*
- series_policies.NASDAQ100U
- series_policies.NASDAQ100U.default_max_loss
- series_policies.NASDAQ100U.per_strike
- series_policies.NASDAQ100U.per_strike.KXNASDAQ100U-*

Referenced in code/docs:
- configs/pal_policy.yaml
- pal_policy.yaml

## configs/pilot.yaml

Keys (nested):
- pilot
- pilot.allowed_series
- pilot.enforce_maker_only
- pilot.max_contracts_per_order
- pilot.max_daily_loss
- pilot.max_unique_bins
- pilot.max_weekly_loss
- pilot.require_acknowledgement
- pilot.require_live_broker
- pilot.session_prefix

Referenced in code/docs:
- configs/pilot.yaml
- pilot.yaml

## configs/portfolio.yaml

Keys (nested):
- factor_vols
- factor_vols.EMPLOYMENT
- factor_vols.INFLATION
- factor_vols.RATES
- factor_vols.WEATHER
- strategy_betas
- strategy_betas.CLAIMS
- strategy_betas.CLAIMS.EMPLOYMENT
- strategy_betas.CLAIMS.INFLATION
- strategy_betas.CPI
- strategy_betas.CPI.INFLATION
- strategy_betas.CPI.RATES
- strategy_betas.TENY
- strategy_betas.TENY.RATES
- strategy_betas.WEATHER
- strategy_betas.WEATHER.WEATHER

Referenced in code/docs:
- configs/portfolio.yaml
- portfolio.yaml

## configs/quality_gates.example.yaml

Keys (nested):
- data_freshness
- data_freshness.max_age_hours
- data_freshness.name
- data_freshness.namespace
- data_freshness.require_et
- data_freshness.timestamp_field
- metrics
- metrics.default
- metrics.default.brier_advantage_min
- metrics.default.crps_advantage_min
- metrics.series
- metrics.series.claims
- metrics.series.claims.brier_advantage_min
- metrics.series.claims.crps_advantage_min
- metrics.series.cpi
- metrics.series.cpi.brier_advantage_min
- metrics.series.cpi.crps_advantage_min
- metrics.series.teny
- metrics.series.teny.brier_advantage_min
- metrics.series.teny.crps_advantage_min
- metrics.series.weather
- metrics.series.weather.brier_advantage_min
- metrics.series.weather.crps_advantage_min
- monitors
- monitors.negative_ev_after_fees
- monitors.non_monotone_ladders
- monitors.tz_not_et
- reconciliation
- reconciliation.dgs_maturity
- reconciliation.name
- reconciliation.namespace
- reconciliation.par_maturity
- reconciliation.tolerance_bps

Referenced in code/docs:
- configs/quality_gates.example.yaml
- quality_gates.example.yaml

## configs/quality_gates.index.yaml

Keys (nested):
- data_freshness
- data_freshness.max_age_seconds
- data_freshness.name
- data_freshness.namespace
- data_freshness.timestamp_field
- monitors
- monitors.tz_not_et

Referenced in code/docs:
- configs/quality_gates.index.yaml
- quality_gates.index.yaml

## configs/quality_gates.yaml

Keys (nested):
- data_freshness
- data_freshness.max_age_hours
- data_freshness.name
- data_freshness.namespace
- data_freshness.require_et
- data_freshness.timestamp_field
- metrics
- metrics.default
- metrics.default.brier_advantage_min
- metrics.default.crps_advantage_min
- metrics.series
- metrics.series.claims
- metrics.series.claims.brier_advantage_min
- metrics.series.claims.crps_advantage_min
- metrics.series.cpi
- metrics.series.cpi.brier_advantage_min
- metrics.series.cpi.crps_advantage_min
- metrics.series.teny
- metrics.series.teny.brier_advantage_min
- metrics.series.teny.crps_advantage_min
- metrics.series.weather
- metrics.series.weather.brier_advantage_min
- metrics.series.weather.crps_advantage_min
- monitors
- monitors.negative_ev_after_fees
- monitors.non_monotone_ladders
- monitors.tz_not_et
- reconciliation
- reconciliation.dgs_maturity
- reconciliation.name
- reconciliation.namespace
- reconciliation.par_maturity
- reconciliation.tolerance_bps

Referenced in code/docs:
- configs/quality_gates.yaml
- quality_gates.yaml

## configs/series.yaml

Keys (nested):
- series
- series.CLAIMS
- series.CLAIMS.notes
- series.CLAIMS.settlement_lag_days
- series.CLAIMS.settlement_source
- series.CPI
- series.CPI.notes
- series.CPI.settlement_lag_days
- series.CPI.settlement_source
- series.TENY
- series.TENY.notes
- series.TENY.settlement_lag_days
- series.TENY.settlement_source
- series.WEATHER
- series.WEATHER.notes
- series.WEATHER.settlement_lag_days
- series.WEATHER.settlement_source

Referenced in code/docs:
- No direct filename/path reference found in scanned source files.

## configs/size_ladder.yaml

Keys (nested):
- current_stage
- stages
- stages.A
- stages.A.description
- stages.A.per_series
- stages.A.per_series.INX
- stages.A.per_series.INX.max_bins
- stages.A.per_series.INX.max_contracts
- stages.A.per_series.INXU
- stages.A.per_series.INXU.max_bins
- stages.A.per_series.INXU.max_contracts
- stages.A.per_series.NASDAQ100
- stages.A.per_series.NASDAQ100.max_bins
- stages.A.per_series.NASDAQ100.max_contracts
- stages.A.per_series.NASDAQ100U
- stages.A.per_series.NASDAQ100U.max_bins
- stages.A.per_series.NASDAQ100U.max_contracts
- stages.B
- stages.B.description
- stages.B.per_series
- stages.B.per_series.INX
- stages.B.per_series.INX.max_bins
- stages.B.per_series.INX.max_contracts
- stages.B.per_series.INXU
- stages.B.per_series.INXU.max_bins
- stages.B.per_series.INXU.max_contracts
- stages.B.per_series.NASDAQ100
- stages.B.per_series.NASDAQ100.max_bins
- stages.B.per_series.NASDAQ100.max_contracts
- stages.B.per_series.NASDAQ100U
- stages.B.per_series.NASDAQ100U.max_bins
- stages.B.per_series.NASDAQ100U.max_contracts
- stages.C
- stages.C.description
- stages.C.per_series
- stages.C.per_series.INX
- stages.C.per_series.INX.max_bins
- stages.C.per_series.INX.max_contracts
- stages.C.per_series.INXU
- stages.C.per_series.INXU.max_bins
- stages.C.per_series.INXU.max_contracts
- stages.C.per_series.NASDAQ100
- stages.C.per_series.NASDAQ100.max_bins
- stages.C.per_series.NASDAQ100.max_contracts
- stages.C.per_series.NASDAQ100U
- stages.C.per_series.NASDAQ100U.max_bins
- stages.C.per_series.NASDAQ100U.max_contracts
- stages.D
- stages.D.description
- stages.D.per_series
- stages.D.per_series.INX
- stages.D.per_series.INX.max_bins
- stages.D.per_series.INX.max_contracts
- stages.D.per_series.INXU
- stages.D.per_series.INXU.max_bins
- stages.D.per_series.INXU.max_contracts
- stages.D.per_series.NASDAQ100
- stages.D.per_series.NASDAQ100.max_bins
- stages.D.per_series.NASDAQ100.max_contracts
- stages.D.per_series.NASDAQ100U
- stages.D.per_series.NASDAQ100U.max_bins
- stages.D.per_series.NASDAQ100U.max_contracts

Referenced in code/docs:
- configs/size_ladder.yaml
- size_ladder.yaml

## configs/strategies/cpi.yaml

Keys (nested):
- blend
- blend.cleveland
- blend.components
- component_weights
- component_weights.autos
- component_weights.gas
- component_weights.shelter
- variance
- variance.component_scale
- version

Referenced in code/docs:
- configs/strategies/cpi.yaml
- cpi.yaml

## configs/systemd/kalshi-alpha-monitors.service

Keys (nested):
- Unable to parse or no structured keys detected.

Referenced in code/docs:
- No direct filename/path reference found in scanned source files.

## configs/systemd/kalshi-alpha-monitors.timer

Keys (nested):
- Unable to parse or no structured keys detected.

Referenced in code/docs:
- No direct filename/path reference found in scanned source files.

## configs/systemd/kalshi-alpha-runner.service

Keys (nested):
- Unable to parse or no structured keys detected.

Referenced in code/docs:
- No direct filename/path reference found in scanned source files.

## configs/systemd/kalshi-alpha-runner.timer

Keys (nested):
- Unable to parse or no structured keys detected.

Referenced in code/docs:
- No direct filename/path reference found in scanned source files.

## configs/systemd/kalshi-alpha-telemetry.service

Keys (nested):
- Unable to parse or no structured keys detected.

Referenced in code/docs:
- No direct filename/path reference found in scanned source files.

## configs/systemd/kalshi-alpha-telemetry.timer

Keys (nested):
- Unable to parse or no structured keys detected.

Referenced in code/docs:
- No direct filename/path reference found in scanned source files.
