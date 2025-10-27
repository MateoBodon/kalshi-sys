"""Generate markdown reports for ladder scans."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from kalshi_alpha.exec.ledger import PaperLedger

if TYPE_CHECKING:  # pragma: no cover
    from kalshi_alpha.exec.runners.scan_ladders import Proposal


def write_markdown_report(
    *,
    series: str,
    proposals: Sequence[Proposal],
    ledger: PaperLedger | None,
    output_dir: Path,
    monitors: dict[str, object] | None = None,
    exposure_summary: dict[str, object] | None = None,
    manifest_path: Path | None = None,
    go_status: bool | None = True,
    fill_alpha: float | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=UTC)
    path = output_dir / f"{ts.strftime('%Y-%m-%d')}.md"
    lines: list[str] = []
    badge_label = "GO" if go_status in {None, True} else "NO-GO"
    badge_emoji = "🟢" if badge_label == "GO" else "🔴"
    lines.append(f"{badge_emoji} **GO/NO-GO:** {badge_label}")
    if manifest_path:
        manifest_str = manifest_path.as_posix() if isinstance(manifest_path, Path) else str(manifest_path)
        lines.append(f"[Archived Manifest]({manifest_str})")
    lines.append("")
    lines.append(f"# {series.upper()} Ladder Scan {ts.strftime('%Y-%m-%d %H:%M UTC')}")
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
    if monitors:
        lines.append("## Monitors")
        for key, value in monitors.items():
            lines.append(f"- {key}: {value}")
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
                    f"{factor}:{float(value):.2f}" for factor, value in sorted(factor_payload.items())
                ) or "-"
                lines.append(
                    f"| {series_name} | {long_side:+d} | {-short_side:+d} | {total_loss:.2f} | {factor_text} |"
                )
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
        fill_pct = max(0.0, min(1.0, record.fill_ratio)) * 100
        lines.append(
            f"| {record.proposal.market_ticker} | {record.proposal.strike:.2f} | "
            f"{record.proposal.side} | {requested} | {expected} | {delta:+d} | {fill_pct:.1f}% |"
        )
    lines.append(f"| **Totals** |  |  | {total_requested} | {total_expected} | {total_delta:+d} |  |")
    lines.append("")
    return lines
