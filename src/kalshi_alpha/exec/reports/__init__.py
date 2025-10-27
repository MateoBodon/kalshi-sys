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
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=UTC)
    path = output_dir / f"{ts.strftime('%Y-%m-%d')}.md"
    lines: list[str] = []
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
