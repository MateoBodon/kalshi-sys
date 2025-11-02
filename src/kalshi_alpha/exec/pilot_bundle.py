"""Assemble a single tarball with key pilot readiness artifacts."""

from __future__ import annotations

import argparse
import io
import json
import tarfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_REPORTS_DIR = Path("reports")
DEFAULT_DATA_ROOT = Path("data")


@dataclass(slots=True)
class BundleItem:
    source: Path
    arcname: Path


def _safe_load_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict):
        return payload
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a pilot review bundle.")
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-telemetry", type=int, default=3, help="Maximum telemetry files to include (default: 3).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(argv) if argv is not None else None)
    reports_dir = args.reports_dir
    data_root = args.data_root
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    output = args.output
    if output is None:
        output = reports_dir / f"pilot_bundle_{timestamp}.tar.gz"
    output.parent.mkdir(parents=True, exist_ok=True)

    items: list[BundleItem] = []
    items.extend(_collect_core_reports(reports_dir))
    items.extend(_collect_monitors(reports_dir))
    items.extend(_collect_scoreboards(reports_dir))
    items.extend(_collect_ladder_reports(reports_dir))
    items.extend(_collect_telemetry(data_root, args.max_telemetry))

    readme_content = _build_pilot_readme(reports_dir)

    manifest_files = [str(item.arcname) for item in items]
    if readme_content:
        manifest_files.append("README_pilot.md")

    manifest: dict[str, object] = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "reports_dir": str(reports_dir),
        "data_root": str(data_root),
        "files": manifest_files,
    }

    with tarfile.open(output, mode="w:gz") as bundle:
        for item in items:
            if not item.source.exists():
                continue
            bundle.add(item.source, arcname=str(item.arcname))
        if readme_content:
            _add_text_file(bundle, "README_pilot.md", readme_content)
        _add_manifest(bundle, manifest)

    print(f"[pilot-bundle] wrote {output}")
    return 0


def _collect_core_reports(reports_dir: Path) -> list[BundleItem]:
    targets = [
        reports_dir / "pilot_ready.json",
        reports_dir / "pilot_readiness.md",
        reports_dir / "_artifacts" / "pilot_session.json",
    ]
    items: list[BundleItem] = []
    for path in targets:
        if path.exists():
            items.append(BundleItem(path, Path("reports") / path.relative_to(reports_dir)))
    return items


def _collect_monitors(reports_dir: Path) -> list[BundleItem]:
    monitor_dir = reports_dir / "_artifacts" / "monitors"
    if not monitor_dir.exists():
        return []
    items: list[BundleItem] = []
    for path in sorted(monitor_dir.glob("*.json")):
        items.append(BundleItem(path, Path("reports/_artifacts/monitors") / path.name))
    go_no_go = reports_dir / "_artifacts" / "go_no_go.json"
    if go_no_go.exists():
        items.append(BundleItem(go_no_go, Path("reports/_artifacts/go_no_go.json")))
    latest_manifest = reports_dir / "_artifacts" / "latest_manifest.txt"
    if latest_manifest.exists():
        items.append(BundleItem(latest_manifest, Path("reports/_artifacts/latest_manifest.txt")))
    return items


def _collect_scoreboards(reports_dir: Path) -> list[BundleItem]:
    items: list[BundleItem] = []
    for path in sorted(reports_dir.glob("scoreboard_*d.md")):
        items.append(BundleItem(path, Path("reports") / path.name))
    pilot_report = reports_dir / "pilot_readiness.md"
    if pilot_report.exists():
        items.append(BundleItem(pilot_report, Path("reports/pilot_readiness.md")))
    return items


def _collect_ladder_reports(reports_dir: Path) -> list[BundleItem]:
    ladder_dir = reports_dir / "ladders"
    if not ladder_dir.exists():
        return []
    items: list[BundleItem] = []
    for path in sorted(ladder_dir.glob("**/*.md")):
        try:
            arcname = Path("reports/ladders") / path.relative_to(ladder_dir)
        except ValueError:
            arcname = Path("reports/ladders") / path.name
        items.append(BundleItem(path, arcname))
    return items


def _collect_telemetry(data_root: Path, limit: int) -> list[BundleItem]:
    telemetry_root = data_root / "raw" / "kalshi"
    if not telemetry_root.exists() or limit <= 0:
        return []
    candidates = sorted(
        telemetry_root.rglob("*.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    selected = candidates[:limit]
    items: list[BundleItem] = []
    for path in selected:
        try:
            rel = path.relative_to(telemetry_root)
        except ValueError:
            rel = Path(path.name)
        arcname = Path("telemetry") / rel
        items.append(BundleItem(path, arcname))
    return items


def _build_pilot_readme(reports_dir: Path) -> str | None:
    policy = _safe_load_json(reports_dir / "pilot_ready.json")
    session = _safe_load_json(reports_dir / "_artifacts" / "pilot_session.json")
    if policy is None and session is None:
        return None

    def _fmt(value: object) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, bool):
            return "OK" if value else "NO"
        if isinstance(value, (int, float)):
            return f"{float(value):.3f}" if abs(float(value)) < 10 else f"{float(value):.1f}"
        return str(value)

    lines: list[str] = ["# Pilot Review Checklist", ""]

    overall = policy.get("overall", {}) if isinstance(policy, dict) else {}
    series_data = policy.get("series") if isinstance(policy, dict) else []
    series_entries = series_data if isinstance(series_data, list) else []
    go_series = [
        str(entry.get("series"))
        for entry in series_entries
        if isinstance(entry, dict) and str(entry.get("recommendation")).upper() == "GO"
    ]
    no_go_series = [
        str(entry.get("series"))
        for entry in series_entries
        if isinstance(entry, dict) and str(entry.get("recommendation")).upper() == "NO_GO"
    ]
    lines.append("## Decision")
    lines.append(f"- GO series: {', '.join(go_series) if go_series else 'none'}")
    lines.append(f"- NO-GO series: {', '.join(no_go_series) if no_go_series else 'none'}")
    raw_reasons = overall.get("global_reasons") if isinstance(overall, dict) else []
    reason_list = [str(reason) for reason in raw_reasons] if isinstance(raw_reasons, list) else []
    if reason_list:
        lines.append("- Global reasons: " + ", ".join(reason_list))
    else:
        lines.append("- Global reasons: none")
    overall_go = all(
        isinstance(entry, dict) and str(entry.get("recommendation")).upper() == "GO"
        for entry in series_entries
    ) and not reason_list
    final_decision = "GO" if overall_go and go_series else "NO-GO"
    rationale_bits: list[str] = []
    if no_go_series:
        rationale_bits.append("NO-GO series: " + ", ".join(no_go_series))
    if reason_list:
        rationale_bits.append("Global reasons: " + ", ".join(reason_list))
    if not rationale_bits:
        rationale_bits.append("All readiness criteria satisfied.")
    lines.append(f"- Final decision: {final_decision}")
    lines.append("- Decision rationale: " + "; ".join(rationale_bits))
    lines.append("")

    lines.append("## Session Snapshot")
    if isinstance(session, dict) and session:
        alerts_raw = session.get("alerts_summary")
        alerts_summary = alerts_raw if isinstance(alerts_raw, dict) else {}
        recent_alerts_raw = alerts_summary.get("recent_alerts")
        recent_alerts = recent_alerts_raw if isinstance(recent_alerts_raw, list) else []
        lines.append(f"- Session ID: {session.get('session_id', 'n/a')}")
        lines.append(f"- Started at: {session.get('started_at', 'n/a')}")
        lines.append(f"- Trades: {session.get('n_trades', 'n/a')}")
        lines.append(f"- Mean Δbps after fees: {_fmt(session.get('mean_delta_bps_after_fees'))}")
        cusum_state = session.get("cusum_state") if isinstance(session, dict) else None
        if cusum_state is None and isinstance(session, dict):
            cusum_state = session.get("cusum_status")
        lines.append(f"- CuSum status: {_fmt(cusum_state)}")
        lines.append(f"- Fill realism gap: {_fmt(session.get('fill_realism_gap'))}")
        lines.append(
            "- Alerts: "
            + (", ".join(str(alert) for alert in recent_alerts) if recent_alerts else "none")
        )
    else:
        lines.append("- pilot_session.json not found")
    lines.append("")

    lines.append("## Checklist")
    ev_flags = overall.get("ev_honesty_flags") if isinstance(overall, dict) else {}
    if isinstance(ev_flags, dict) and ev_flags:
        summaries: list[str] = []
        for series, bins in ev_flags.items():
            if not isinstance(bins, list):
                continue
            parts: list[str] = []
            for entry in bins:
                if not isinstance(entry, dict):
                    continue
                label = f"{entry.get('side')} {entry.get('strike')}"
                weight = entry.get("recommended_weight")
                cap = entry.get("recommended_cap")
                annotations: list[str] = []
                if weight is not None:
                    annotations.append(f"w={_fmt(weight)}")
                if cap is not None:
                    annotations.append(f"cap={_fmt(cap)}")
                if annotations:
                    label += " (" + ", ".join(annotations) + ")"
                parts.append(label)
            if parts:
                summaries.append(f"{series}: " + ", ".join(parts))
        lines.append("- EV honesty: flagged — " + " | ".join(summaries))
    else:
        lines.append("- EV honesty: clean")

    monitors_summary = policy.get("monitors_summary", {}) if isinstance(policy, dict) else {}
    statuses = monitors_summary.get("statuses") if isinstance(monitors_summary, dict) else {}
    if not isinstance(statuses, dict):
        statuses = {}
    seq_status = statuses.get("ev_seq_guard")
    if seq_status is None and isinstance(session, dict):
        seq_status = session.get("cusum_state") or session.get("cusum_status")
    lines.append(f"- Sequential guard (CuSum): {_fmt(seq_status)}")

    freeze_series = overall.get("freeze_violation_series") if isinstance(overall, dict) else []
    if isinstance(freeze_series, list) and freeze_series:
        lines.append("- Freeze violations: " + ", ".join(str(series) for series in freeze_series))
    else:
        lines.append("- Freeze violations: none")

    drawdown = policy.get("drawdown") if isinstance(policy, dict) else {}
    if isinstance(drawdown, dict):
        lines.append(f"- Drawdown: {'OK' if drawdown.get('ok') else 'BREACHED'}")
    else:
        lines.append("- Drawdown: n/a")

    ws_status = statuses.get("ws_disconnects")
    auth_status = statuses.get("auth_errors")
    lines.append(f"- WS health: {ws_status or 'n/a'}; Auth health: {auth_status or 'n/a'}")

    freshness = policy.get("freshness", {}) if isinstance(policy, dict) else {}
    if isinstance(freshness, dict):
        ledger_age = freshness.get("ledger_age_minutes", freshness.get("ledger_minutes"))
        ledger_limit = freshness.get("ledger_threshold_minutes")
        ledger_state = (
            "STALE"
            if ledger_age is not None and ledger_limit is not None and ledger_age > ledger_limit
            else "OK"
        )
        lines.append(
            f"- Ledger staleness: {_fmt(ledger_age)} min (limit {_fmt(ledger_limit)}) — {ledger_state}"
        )
        monitor_age = freshness.get("monitors_age_minutes", freshness.get("monitors_minutes"))
        monitor_limit = freshness.get("monitors_threshold_minutes")
        monitor_state = (
            "STALE"
            if monitor_age is not None and monitor_limit is not None and monitor_age > monitor_limit
            else "OK"
        )
        lines.append(
            f"- Monitor staleness: {_fmt(monitor_age)} min (limit {_fmt(monitor_limit)}) — {monitor_state}"
        )
    else:
        lines.append("- Staleness: n/a")

    lines.append("")
    return "\n".join(lines)


def _add_text_file(bundle: tarfile.TarFile, name: str, content: str) -> None:
    payload = content.encode("utf-8")
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mtime = time.time()
    bundle.addfile(info, io.BytesIO(payload))


def _add_manifest(bundle: tarfile.TarFile, manifest: dict[str, object]) -> None:
    payload = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    info = tarfile.TarInfo(name="manifest.json")
    info.size = len(payload)
    info.mtime = time.time()
    bundle.addfile(info, io.BytesIO(payload))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
