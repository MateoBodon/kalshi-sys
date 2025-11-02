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

    manifest: dict[str, object] = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "reports_dir": str(reports_dir),
        "data_root": str(data_root),
        "files": [str(item.arcname) for item in items],
    }

    with tarfile.open(output, mode="w:gz") as bundle:
        for item in items:
            if not item.source.exists():
                continue
            bundle.add(item.source, arcname=str(item.arcname))
        _add_manifest(bundle, manifest)

    print(f"[pilot-bundle] wrote {output}")
    return 0


def _collect_core_reports(reports_dir: Path) -> list[BundleItem]:
    targets = [
        reports_dir / "pilot_ready.json",
        reports_dir / "pilot_readiness.md",
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


def _add_manifest(bundle: tarfile.TarFile, manifest: dict[str, object]) -> None:
    payload = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
    info = tarfile.TarInfo(name="manifest.json")
    info.size = len(payload)
    info.mtime = time.time()
    bundle.addfile(info, io.BytesIO(payload))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
