"""Generate markdown reports for ladder scans."""

from __future__ import annotations

import json
import math
import statistics
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from kalshi_alpha.exec.ledger import ExecutionMetrics, PaperLedger

GO_ARTIFACT_PATH = Path("reports/_artifacts/go_no_go.json")


def _float_metric(metrics: ExecutionMetrics, key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _int_metric(metrics: ExecutionMetrics, key: str) -> int:
    value = metrics.get(key)
    if isinstance(value, (int, float)):
        return int(value)
    return 0


def _resolve_latest_go_artifact(candidate: Path) -> Path | None:
    """Return the most recent go/no-go artifact matching ``candidate``."""

    path = Path(candidate)
    if path.exists():
        return path

    directory = path.parent
    if not directory.exists():
        return None

    pattern = f"{path.stem}*.json"
    latest_path: Path | None = None
    latest_mtime: float | None = None
    for entry in sorted(directory.glob(pattern)):
        try:
            mtime = entry.stat().st_mtime
        except OSError:  # pragma: no cover - filesystem race
            continue
        if latest_mtime is None or mtime >= latest_mtime:
            latest_path = entry
            latest_mtime = mtime
    return latest_path


def _load_go_status(artifact_path: Path) -> bool | None:
    """Return GO/NO-GO boolean from artifact, or ``None`` if unavailable."""

    resolved = _resolve_latest_go_artifact(artifact_path)
    if resolved is None:
        return None
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive
        return None
    go_value = payload.get("go") if isinstance(payload, dict) else None
    if isinstance(go_value, bool):
        return go_value
    if isinstance(go_value, (int, float)):
        return bool(go_value)
    return None

if TYPE_CHECKING:  # pragma: no cover
    from kalshi_alpha.exec.runners.scan_ladders import Proposal


def write_markdown_report(  # noqa: PLR0913, PLR0912, PLR0915
    *,
    series: str,
    proposals: Sequence[Proposal],
    ledger: PaperLedger | None,
    output_dir: Path,
    monitors: dict[str, object] | None = None,
    exposure_summary: dict[str, object] | None = None,
    manifest_path: Path | None = None,
    go_status: bool | None = None,
    go_artifact_path: Path | None = GO_ARTIFACT_PATH,
    fill_alpha: float | None = None,
    mispricings: Sequence[dict[str, object]] | None = None,
    model_metadata: dict[str, object] | None = None,
    scorecard_summary: Sequence[dict[str, object]] | None = None,
    outstanding_summary: dict[str, int] | None = None,
    pilot_metadata: dict[str, object] | None = None,
    execution_metrics: ExecutionMetrics | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=UTC)
    path = output_dir / f"{ts.strftime('%Y-%m-%d')}.md"
    lines: list[str] = []
    artifact_go = _load_go_status(go_artifact_path) if go_artifact_path else None
    effective_go = artifact_go if artifact_go is not None else go_status
    if effective_go is None:
        effective_go = True
    badge_label = "GO" if effective_go else "NO-GO"
    badge_emoji = "🟢" if badge_label == "GO" else "🔴"
    lines.append(f"{badge_emoji} **GO/NO-GO:** {badge_label}")
    if outstanding_summary is not None:
        total = sum(outstanding_summary.values())
        breakdown = ", ".join(
            f"{mode}={count}" for mode, count in sorted(outstanding_summary.items())
        )
        lines.append(f"Outstanding orders: {total} ({breakdown})")
    throttle_rows = 0
    if ledger:
        throttle_rows = sum(
            1
            for record in ledger.records
            if getattr(record, "size_throttled", False)
        )
    if throttle_rows:
        lines.append(f"Size throttle active on {throttle_rows} rows")
    if pilot_metadata:
        if pilot_metadata.get("force_run"):
            lines.append("FORCE-RUN (DRY)")
        window_label = pilot_metadata.get("window_et")
        if window_label:
            lines.append(f"Scheduled window (ET): {window_label}")
        lines.append(_format_pilot_header(pilot_metadata))
    if manifest_path:
        if isinstance(manifest_path, Path):
            manifest_str = manifest_path.as_posix()
        else:
            manifest_str = str(manifest_path)
        lines.append(f"[Archived Manifest]({manifest_str})")
    lines.append("")
    lines.append(f"# {series.upper()} Ladder Scan {ts.strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")
    if model_metadata:
        lines.append("## Model Configuration")
        version = model_metadata.get("model_version") if isinstance(model_metadata, dict) else None
        if version is not None:
            lines.append(f"- Model Version: {version}")
        component_weights = None
        if isinstance(model_metadata, dict):
            component_weights = model_metadata.get("component_weights")
        if isinstance(component_weights, dict) and component_weights:
            lines.append("### CPI Component Weights")
            lines.append("| Component | Weight |")
            lines.append("| --- | --- |")
            for key in sorted(component_weights):
                value = component_weights[key]
                try:
                    formatted = f"{float(value):.3f}"
                except (TypeError, ValueError):
                    formatted = str(value)
                lines.append(f"| {key} | {formatted} |")
            lines.append("")
        other_metadata = {
            key: value
            for key, value in (model_metadata.items() if isinstance(model_metadata, dict) else [])
            if key not in {"model_version", "component_weights"}
        }
        for key, value in sorted(other_metadata.items()):
            lines.append(f"- {key}: {value}")
        lines.append("")
    if scorecard_summary is not None:
        lines.append("## Replay Scorecard")
        if not scorecard_summary:
            lines.append("No replay scorecard data available.")
        else:
            top_records = sorted(
                scorecard_summary,
                key=lambda rec: float(rec.get("mean_abs_cdf_delta", 0.0)),
                reverse=True,
            )
            lines.append("| Market | Mean | Max | Prob Gap | Max Kink | Kinks |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for record in top_records[:5]:
                market_name = record.get("market_ticker", "-")
                mean_delta = float(record.get("mean_abs_cdf_delta", 0.0))
                max_delta = float(record.get("max_abs_cdf_delta", 0.0))
                prob_gap = float(record.get("prob_sum_gap", 0.0))
                max_kink = float(record.get("max_kink", 0.0))
                kink_count = int(record.get("kink_count", 0))
                row_line = (
                    f"| {market_name} | {mean_delta:.4f} | {max_delta:.4f} | "
                    f"{prob_gap:.4f} | {max_kink:.4f} | {kink_count} |"
                )
                lines.append(row_line)
            lines.append("")
            lines.append(f"Markets evaluated: {len(scorecard_summary)}")
        lines.append("")
    lines.append("## Proposals")
    lines.append(
        "| Strike | Side | Contracts | Maker EV | Max Loss | Strategy S(x) | Market S(x) |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for proposal in proposals:
        row = (
            f"| {proposal.strike:.2f} | {proposal.side} | {proposal.contracts} | "
            f"{proposal.maker_ev:.2f} | {proposal.max_loss:.2f} | "
            f"{proposal.strategy_probability:.3f} | {proposal.survival_market:.3f} |"
        )
        lines.append(row)
    lines.append("")
    if ledger:
        summary = ledger.to_dict()
        lines.append("## Paper Ledger Summary")
        lines.append(f"Expected PnL: {summary['expected_pnl']:.2f} USD")
        lines.append(f"Max Loss: {summary['max_loss']:.2f} USD")
        lines.append(f"Trades: {summary['trades']}")
        lines.append("")
        lines.extend(_expected_vs_realized_rows(ledger))
        if monitors is not None and throttle_rows:
            monitors.setdefault("size_throttled_rows", throttle_rows)
    ev_table_data = None
    ev_max_delta = None
    if monitors:
        table_candidate = monitors.get("ev_honesty_table")
        if isinstance(table_candidate, list) and table_candidate:
            ev_table_data = table_candidate
        max_delta_candidate = monitors.get("ev_honesty_max_delta")
        if max_delta_candidate is not None:
            try:
                ev_max_delta = float(max_delta_candidate)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                ev_max_delta = None

    if ev_table_data or ledger:
        ev_lines, ev_diff = _ev_honesty_rows(
            ledger,
            table_data=ev_table_data,
            max_delta=ev_max_delta,
        )
        if ev_lines:
            lines.extend(ev_lines)
            lines.append("")
            if monitors is not None and ev_diff is not None:
                monitors.setdefault("ev_per_contract_diff_max", ev_diff)

    if monitors:
        lines.append("## Monitors")
        for key, value in monitors.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    if mispricings:
        lines.append("## Mispricing & Kinks")
        lines.append("| Market | Prob Gap | Max Kink | Legs | Direction | Delta Sum |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for record in mispricings:
            spreads = record.get("spreads") or []
            top = spreads[0] if spreads else None
            legs = top["legs"] if top else 0
            direction = top["direction"] if top else "-"
            delta_sum = top["delta_sum"] if top else 0.0
            row_line = (
                f"| {record['market_ticker']} | {record['prob_sum_gap']:.4f} | "
                f"{record['max_kink']:.4f} | {legs} | {direction} | {delta_sum:+.4f} |"
            )
            lines.append(row_line)
        lines.append("")
    if exposure_summary:
        lines.append("## Portfolio Exposure")
        total_loss = exposure_summary.get("total_max_loss")
        if total_loss is not None:
            lines.append(f"- Total Max Loss: {float(total_loss):.2f} USD")
        var_value = exposure_summary.get("var")
        if var_value is not None:
            lines.append(f"- Portfolio VaR: {float(var_value):.2f} USD")
        if fill_alpha is not None:
            lines.append(f"- Fill Alpha: {float(fill_alpha):.2f}")
        factors = exposure_summary.get("factors") or {}
        if factors:
            lines.append("")
            lines.append("| Factor | Exposure |")
            lines.append("| --- | --- |")
            for factor, value in sorted(factors.items()):
                lines.append(f"| {factor} | {float(value):.2f} |")
        per_series = exposure_summary.get("per_series") or {}
        if per_series:
            lines.append("")
            lines.append("| Series | Max Loss |")
            lines.append("| --- | --- |")
            for name, value in sorted(per_series.items()):
                lines.append(f"| {name} | {float(value):.2f} |")
        net_contracts = exposure_summary.get("net_contracts") or {}
        if net_contracts:
            lines.append("")
            lines.append("| Market | Net Contracts |")
            lines.append("| --- | --- |")
            for market, value in sorted(net_contracts.items()):
                lines.append(f"| {market} | {int(value):+d} |")
        lines.append("")
        market_losses = exposure_summary.get("market_losses") or {}
        market_series = exposure_summary.get("market_series") or {}
        if market_losses and exposure_summary.get("net_contracts"):
            lines.append("## Net Ladder Exposure")
            lines.append("| Series | Market | Net Contracts | Max Loss |")
            lines.append("| --- | --- | --- | --- |")
            for market, net in sorted((exposure_summary.get("net_contracts") or {}).items()):
                series_name = market_series.get(market, "")
                max_loss = float(market_losses.get(market, 0.0))
                lines.append(f"| {series_name} | {market} | {int(net):+d} | {max_loss:.2f} |")
            lines.append("")
        series_net = exposure_summary.get("series_net") or {}
        series_factors = exposure_summary.get("series_factors") or {}
        if series_net:
            lines.append("### Series Summary")
            lines.append("| Series | Net Long | Net Short | Total Max Loss | Factor Loads |")
            lines.append("| --- | --- | --- | --- | --- |")
            for series_name, payload in sorted(series_net.items()):
                long_side = int(payload.get("long", 0))
                short_side = int(payload.get("short", 0))
                total_loss = float(per_series.get(series_name, 0.0))
                factor_payload = series_factors.get(series_name, {})
                factor_text = ", ".join(
                    f"{factor}:{float(value):.2f}"
                    for factor, value in sorted(factor_payload.items())
                ) or "-"
                row_line = (
                    f"| {series_name} | {long_side:+d} | {-short_side:+d} | "
                    f"{total_loss:.2f} | {factor_text} |"
                )
                lines.append(row_line)
            lines.append("")
    if execution_metrics:
        lines.append("## Fill & Slippage")
        records = _int_metric(execution_metrics, "records")
        lines.append(f"- Samples: {records}")
        fill_ratio_value = _float_metric(execution_metrics, "fill_ratio_avg")
        if fill_ratio_value is not None:
            lines.append(f"- Fill Ratio: {fill_ratio_value:.3f}")
        alpha_value = _float_metric(execution_metrics, "alpha_target_avg")
        if alpha_value is not None:
            lines.append(f"- Alpha Target: {alpha_value:.3f}")
        delta_value = _float_metric(execution_metrics, "fill_ratio_minus_alpha")
        if delta_value is not None:
            lines.append(f"- Fill - Alpha: {delta_value:+.3f}")
        slippage_ticks = _float_metric(execution_metrics, "slippage_ticks_avg")
        if slippage_ticks is not None:
            lines.append(f"- Avg Slippage (ticks): {slippage_ticks:.3f}")
        slippage_usd = _float_metric(execution_metrics, "slippage_usd_avg")
        if slippage_usd is not None:
            lines.append(f"- Avg Slippage (USD): {slippage_usd:.4f}")
        expected_bps = _float_metric(execution_metrics, "ev_expected_bps_avg")
        if expected_bps is not None:
            lines.append(f"- Expected EV (bps): {expected_bps:.2f}")
        realized_bps = _float_metric(execution_metrics, "ev_realized_bps_avg")
        if realized_bps is not None:
            lines.append(f"- Simulated EV (bps): {realized_bps:.2f}")
        lines.append("")
    lines.append("## Proposal Details")
    for proposal in proposals:
        lines.append(f"### Strike {proposal.strike:.2f}")
        for key, value in asdict(proposal).items():
            if key == "metadata" and value is not None:
                lines.append("- metadata:")
                for meta_key, meta_value in value.items():
                    lines.append(f"  - {meta_key}: {meta_value}")
            else:
                lines.append(f"- {key}: {value}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _expected_vs_realized_rows(ledger: PaperLedger) -> list[str]:
    if not ledger.records:
        return []
    total_requested = sum(record.proposal.contracts for record in ledger.records)
    total_expected = sum(record.expected_fills for record in ledger.records)
    total_delta = total_expected - total_requested
    lines = [
        "### Expected vs Realized Fills",
        "| Market | Bin | Side | Requested | Expected | Delta | Fill % |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in ledger.records:
        requested = record.proposal.contracts
        expected = record.expected_fills
        delta = expected - requested
        fill_pct = max(0.0, min(1.0, record.fill_ratio_realized)) * 100
        lines.append(
            f"| {record.proposal.market_ticker} | {record.proposal.strike:.2f} | "
            f"{record.proposal.side} | {requested} | {expected} | {delta:+d} | {fill_pct:.1f}% |"
        )
    totals_line = (
        f"| **Totals** |  |  | {total_requested} | {total_expected} | {total_delta:+d} |  |"
    )
    lines.append(totals_line)
    lines.append("")

    ev_expected_values: list[float] = []
    ev_realized_values: list[float] = []
    delta_values: list[float] = []
    lines.append("### Expected vs Realized EV (bps)")
    lines.append("| Market | Bin | Side | Expected (bps) | Realized (bps) | Δ (bps) |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for record in ledger.records:
        expected_bps = float(getattr(record, "ev_expected_bps", 0.0))
        realized_bps = float(getattr(record, "ev_realized_bps", 0.0))
        delta_bps = realized_bps - expected_bps
        ev_expected_values.append(expected_bps)
        ev_realized_values.append(realized_bps)
        delta_values.append(delta_bps)
        ev_line = (
            f"| {record.proposal.market_ticker} | {record.proposal.strike:.2f} | "
            f"{record.proposal.side} | {expected_bps:+.1f} | {realized_bps:+.1f} | "
            f"{delta_bps:+.1f} |"
        )
        lines.append(ev_line)
    sample_size = len(ev_expected_values)
    if sample_size:
        mean_expected = statistics.mean(ev_expected_values)
        mean_realized = statistics.mean(ev_realized_values)
        mean_delta = statistics.mean(delta_values)
        std_delta = statistics.stdev(delta_values) if sample_size > 1 else 0.0
        if sample_size > 1:
            t_stat = mean_delta / (std_delta / math.sqrt(sample_size)) if std_delta > 0 else 0.0
        else:
            t_stat = 0.0
        badge = _confidence_badge(sample_size, t_stat)
        lines.append(
            f"| **Mean** |  |  | {mean_expected:+.1f} | {mean_realized:+.1f} | {mean_delta:+.1f} |"
        )
        lines.append("")
        lines.append(f"Sample Size: {sample_size} trades")
        lines.append(f"Confidence: {badge} (t={t_stat:.2f})")
        lines.extend(_ev_plot_lines(mean_expected, mean_realized))
        lines.append("")
    return lines


def _confidence_badge(sample_size: int, t_stat: float) -> str:
    if sample_size >= 200 and t_stat >= 2.0:
        return "✓"
    if t_stat >= 1.0:
        return "△"
    return "✗"


def _ev_plot_lines(expected: float, realized: float) -> list[str]:
    scale = max(abs(expected), abs(realized), 1.0)

    def _bar(label: str, value: float) -> str:
        proportion = min(1.0, abs(value) / scale)
        length = max(1, int(round(proportion * 20)))
        bar = "█" * length
        sign = "+" if value >= 0 else "-"
        return f"{label:<9}: {sign}{bar:<20} {value:.1f} bps"

    return ["```", _bar("expected", expected), _bar("realized", realized), "```"]


def _ev_honesty_rows(  # noqa: PLR0911, PLR0912, PLR0915
    ledger: PaperLedger | None,
    *,
    table_data: Sequence[dict[str, object]] | None = None,
    max_delta: float | None = None,
) -> tuple[list[str], float | None]:
    def _safe_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    if table_data:
        header_row = (
            "| Market | Strike | EV_per_contract_original | EV_per_contract_replay | "
            "EV_total_original | EV_total_replay | Delta |"
        )
        header = [
            "### EV Honesty",
            header_row,
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        rows: list[str] = []
        local_max = 0.0
        for entry in table_data:
            market = entry.get("market_ticker", "-")
            strike = _safe_float(entry.get("strike"))
            ev_orig_pc = _safe_float(entry.get("maker_ev_per_contract_original"))
            ev_replay_pc = _safe_float(entry.get("maker_ev_per_contract_replay"))
            ev_orig_total = _safe_float(entry.get("maker_ev_original"))
            ev_replay_total = _safe_float(entry.get("maker_ev_replay"))
            delta = abs(ev_orig_pc - ev_replay_pc)
            local_max = max(local_max, delta)
            rows.append(
                f"| {market} | {strike:.2f} | {ev_orig_pc:.2f} | {ev_replay_pc:.2f} | "
                f"{ev_orig_total:.2f} | {ev_replay_total:.2f} | {delta:.2f} |"
            )
        summary: list[str] = []
        resolved_max = max_delta if max_delta is not None else local_max
        summary.append("")
        summary.append(f"Max per-contract delta: {resolved_max:.2f}")
        summary.append("")
        return header + rows + summary, resolved_max

    if ledger is None or not ledger.records:
        return [], None
    replay_path = Path("reports/_artifacts/replay_ev.parquet")
    if not replay_path.exists():
        return [], None
    try:
        replay_df = pl.read_parquet(replay_path)
    except Exception:
        return [], None
    if replay_df.is_empty():
        return [], None
    if {"market_id", "strike"}.difference(replay_df.columns):
        return [], None
    lookup = {}
    for row in replay_df.to_dicts():
        market_id = row.get("market_id")
        strike_value = row.get("strike")
        if market_id is None or strike_value is None:
            continue
        try:
            key = (str(market_id), float(strike_value))
        except (TypeError, ValueError):
            continue
        lookup[key] = row
    header = [
        "### EV Honesty",
        (
            "| Market | Strike | EV_per_contract_original | EV_per_contract_replay | "
            "EV_total_original | EV_total_replay | Delta |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    rows = []
    max_diff = 0.0
    for record in ledger.records:
        key = (record.proposal.market_id, float(record.proposal.strike))
        replay_row = lookup.get(key)
        if replay_row is None:
            continue
        per_contract_original = float(record.proposal.maker_ev_per_contract)
        total_original = float(record.expected_value)
        per_contract_replay = _safe_float(
            replay_row.get("maker_ev_per_contract_replay", replay_row.get("maker_ev_replay"))
        )
        total_replay = _safe_float(replay_row.get("maker_ev_replay"))
        delta = abs(per_contract_original - per_contract_replay)
        rows.append(
            f"| {record.proposal.market_ticker} | {record.proposal.strike:.2f} | "
            f"{per_contract_original:.2f} | {per_contract_replay:.2f} | "
            f"{total_original:.2f} | {total_replay:.2f} | {delta:.2f} |"
        )
        max_diff = max(max_diff, delta)
    if not rows:
        return [], None
    rows.append("")
    rows.append(f"Max per-contract delta: {max_diff:.2f}")
    rows.append("")
    return header + rows, max_diff


def _format_pilot_header(metadata: dict[str, object]) -> str:
    parts: list[str] = []
    mode = metadata.get("mode")
    if mode:
        parts.append(f"mode={str(mode).lower()}")
    kelly = metadata.get("kelly_cap")
    if isinstance(kelly, (int, float)):
        parts.append(f"kelly_cap={kelly:.2f}")
    max_var = metadata.get("max_var")
    if isinstance(max_var, (int, float)):
        parts.append(f"max_var={max_var:.2f}")
    fill_alpha = metadata.get("fill_alpha")
    if isinstance(fill_alpha, (int, float)):
        parts.append(f"α={fill_alpha:.3f}")
    outstanding = metadata.get("outstanding_total")
    if isinstance(outstanding, (int, float)):
        parts.append(f"outstanding={int(outstanding)}")
    return "Live Pilot: " + " | ".join(parts)
