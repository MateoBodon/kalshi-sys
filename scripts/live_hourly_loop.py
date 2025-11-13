#!/usr/bin/env python3
"""Continuous runner for INX/NDX hourly scans with multiple parameter sets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

from kalshi_alpha.config import size_ladder as size_ladder_config
from kalshi_alpha.sched import windows as sched_windows

ET_ZONE = ZoneInfo("America/New_York")
SIZE_LADDER_PATH = Path("configs/size_ladder.yaml")
DISCOVERY_PATH = Path("reports/_artifacts/discovery_today.json")
SLO_METRICS_PATH = Path("reports/_artifacts/monitors/slo_selfcheck.json")
DEFAULT_STATE_ARTIFACT = Path("reports/_artifacts/live_loop_state.json")
FRESHNESS_THRESHOLD_MS = 700.0
_DISCOVERY_CACHE: dict[str, object] = {"mtime": None, "target": None}


def _next_target_hour(now_utc: datetime) -> int:
    now_et = now_utc.astimezone(ET_ZONE)
    hour = now_et.hour
    if now_et.minute >= 55:
        hour = (hour + 1) % 24
    return hour


def _run_commands(commands: Sequence[Sequence[str]], target_hour: int, dry: bool = False) -> None:
    env = dict(os.environ)
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    existing = env.get("PYTHONPATH", "")
    parts = [str(src_path)]
    if existing:
        parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(parts)

    for base_cmd in commands:
        cmd = [*base_cmd, "--target-hour", str(target_hour)]
        print(f"[live-loop] running: {' '.join(cmd)}", flush=True)
        if dry:
            continue
        try:
            subprocess.run(cmd, check=False, env=env)
        except KeyboardInterrupt:  # pragma: no cover - handled by outer loop
            raise
        except Exception as exc:  # pragma: no cover - best-effort logging
            print(f"[live-loop] ERROR executing {' '.join(cmd)} -> {exc}", file=sys.stderr, flush=True)


def _base_command(
    *,
    series: str,
    contracts: int,
    impact_cap: float,
    pilot_config: str | None = None,
) -> list[str]:
    cmd: list[str] = [
        sys.executable,
        "-m",
        "kalshi_alpha.exec.runners.scan_ladders",
        "--series",
        series,
        "--online",
        "--pilot",
        "--contracts",
        str(max(1, contracts)),
        "--min-ev",
        "0.0",
        "--maker-only",
        "--impact-cap",
        f"{impact_cap:.4f}",
        "--report",
        "--paper-ledger",
        "--broker",
        "live",
        "--daily-loss-cap",
        "50",
        "--weekly-loss-cap",
        "250",
        "--i-understand-the-risks",
        "--force-gate-pass",
    ]
    if pilot_config:
        cmd.extend(["--pilot-config", pilot_config])
    return cmd


def build_hourly_command_sets(
    stage: size_ladder_config.SizeStage | None,
) -> list[list[str]]:
    """Return the size-aware list of scanner invocations per loop."""

    if stage is None:
        return [
            _base_command(series="INXU", contracts=1, impact_cap=0.05, pilot_config="configs/pilot_bins4.yaml"),
            _base_command(series="INXU", contracts=2, impact_cap=0.02),
            _base_command(series="NASDAQ100U", contracts=1, impact_cap=0.05, pilot_config="configs/pilot_bins4.yaml"),
            _base_command(series="NASDAQ100U", contracts=2, impact_cap=0.02),
        ]

    commands: list[list[str]] = []
    for series in ("INXU", "NASDAQ100U"):
        limits = stage.limits_for(series)
        if limits is None or limits.max_contracts <= 0:
            continue
        pilot_path = _pilot_for_bins(limits.max_bins)
        wide_contracts = min(1, limits.max_contracts)
        if wide_contracts > 0:
            commands.append(
                _base_command(
                    series=series,
                    contracts=wide_contracts,
                    impact_cap=0.05,
                    pilot_config=pilot_path,
                )
            )
        if limits.max_contracts > wide_contracts:
            commands.append(
                _base_command(
                    series=series,
                    contracts=limits.max_contracts,
                    impact_cap=0.02,
                    pilot_config=pilot_path,
                )
            )
    return commands


def build_close_command_sets(stage: size_ladder_config.SizeStage | None) -> list[list[str]]:
    """Return commands specifically for the 16:00 ET close window."""

    if stage is None:
        return [
            _base_command(series="INX", contracts=2, impact_cap=0.02),
            _base_command(series="NASDAQ100", contracts=2, impact_cap=0.02),
        ]
    commands: list[list[str]] = []
    for series in ("INX", "NASDAQ100"):
        limits = stage.limits_for(series)
        if limits is None or limits.max_contracts <= 0:
            continue
        commands.append(
            _base_command(
                series=series,
                contracts=max(1, limits.max_contracts),
                impact_cap=0.02,
                pilot_config=_pilot_for_bins(limits.max_bins),
            )
        )
    return commands


def _pilot_for_bins(max_bins: int) -> str | None:
    if max_bins >= 4:
        return "configs/pilot_bins4.yaml"
    return None


def _load_size_stage() -> size_ladder_config.SizeStage | None:
    try:
        ladder = size_ladder_config.load_size_ladder(SIZE_LADDER_PATH)
        stage = ladder.stage()
        print(f"[live-loop] size stage {stage.name}: {stage.description}")
        return stage
    except Exception as exc:  # pragma: no cover - config errors fall back to defaults
        print(f"[live-loop] size ladder unavailable ({exc}); using legacy defaults")
        return None


def _close_window(now_et: datetime) -> sched_windows.TradingWindow | None:
    for window in sched_windows.windows_for_day(now_et.date()):
        if window.target_type == "close":
            return window
    return None


def _discovery_close_target() -> datetime | None:
    try:
        stat = DISCOVERY_PATH.stat()
    except FileNotFoundError:
        _DISCOVERY_CACHE["target"] = None
        _DISCOVERY_CACHE["mtime"] = None
        return None
    cached_mtime = _DISCOVERY_CACHE.get("mtime")
    if cached_mtime == stat.st_mtime:
        return _DISCOVERY_CACHE.get("target")
    try:
        payload = json.loads(DISCOVERY_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _DISCOVERY_CACHE["target"] = None
        _DISCOVERY_CACHE["mtime"] = stat.st_mtime
        return None
    target: datetime | None = None
    for window in payload.get("windows", []):
        if not isinstance(window, dict):
            continue
        if str(window.get("target_type")).lower() != "close":
            continue
        target_et = window.get("target_et")
        if target_et:
            try:
                target = datetime.fromisoformat(str(target_et))
            except ValueError:
                target = None
            break
    _DISCOVERY_CACHE["target"] = target
    _DISCOVERY_CACHE["mtime"] = stat.st_mtime
    return target


def _close_phase(
    window: sched_windows.TradingWindow,
    now_et: datetime,
    discovery_target: datetime | None,
) -> dict[str, object]:
    target_et = discovery_target.astimezone(ET_ZONE) if discovery_target else window.target_et
    cancel_buffer = window.target_et - window.freeze_et
    freeze_et = target_et - cancel_buffer
    prep_cutoff = max(window.start_et, target_et - timedelta(minutes=5))
    freshness_strict = max(window.start_et, target_et - timedelta(minutes=1))
    if now_et < window.start_et:
        state = "IDLE"
    elif now_et < prep_cutoff:
        state = "PREP"
    elif now_et < freeze_et:
        state = "MAKE"
    else:
        state = "FREEZE"
    return {
        "state": state,
        "target_et": target_et,
        "freeze_et": freeze_et,
        "freshness_strict_et": freshness_strict,
    }


def _freshness_breach(threshold_ms: float = FRESHNESS_THRESHOLD_MS) -> dict[str, float]:
    try:
        lines = SLO_METRICS_PATH.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return {}
    breaches: dict[str, float] = {}
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        series = str(payload.get("series") or "").upper()
        freshness = payload.get("freshness_p95_ms")
        if series and isinstance(freshness, (int, float)) and freshness > threshold_ms:
            breaches[series] = float(freshness)
    return breaches


def _write_state_artifact(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _close_gate(now_et: datetime) -> dict[str, object]:
    window = _close_window(now_et)
    if window is None:
        return {
            "state": "IDLE",
            "go": False,
            "reasons": ["no_close_window"],
            "target_et": None,
            "freshness_strict_et": None,
        }
    discovery_target = _discovery_close_target()
    phase = _close_phase(window, now_et, discovery_target)
    reasons: list[str] = []
    go = phase["state"] in {"PREP", "MAKE"}
    if not go:
        reasons.append(f"state_{phase['state'].lower()}")
    else:
        freshness_gate_ts = phase["freshness_strict_et"]
        if freshness_gate_ts and now_et >= freshness_gate_ts:
            breaches = _freshness_breach()
            if breaches:
                go = False
                series_list = ",".join(sorted(breaches))
                reasons.append(f"freshness>{FRESHNESS_THRESHOLD_MS:.0f}ms:{series_list}")
    return {
        "state": phase["state"],
        "go": go,
        "reasons": reasons,
        "target_et": phase["target_et"],
        "freshness_strict_et": phase["freshness_strict_et"],
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Continuously run hourly INX/NDX live scans.")
    parser.add_argument("--interval", type=float, default=180.0, help="Sleep seconds between loops (default: 180).")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument(
        "--once", action="store_true", help="Run a single loop (useful for testing) instead of continuous service."
    )
    parser.add_argument(
        "--state-artifact",
        type=Path,
        default=DEFAULT_STATE_ARTIFACT,
        help="Path to write GO/NO-GO state JSON (default: reports/_artifacts/live_loop_state.json)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    stage = _load_size_stage()
    hourly_commands = build_hourly_command_sets(stage)
    close_commands = build_close_command_sets(stage)

    try:
        while True:
            now_utc = datetime.now(tz=UTC)
            now_et = now_utc.astimezone(ET_ZONE)
            target_hour = _next_target_hour(now_utc)
            _run_commands(hourly_commands, target_hour, dry=args.dry_run)
            close_info = _close_gate(now_et)
            if close_commands and close_info["state"] in {"PREP", "MAKE"}:
                if close_info["go"]:
                    _run_commands(close_commands, 16, dry=args.dry_run)
                else:
                    reasons = ",".join(close_info["reasons"] or [])
                    print(f"[live-loop] skipping close run ({reasons or 'gated'})", flush=True)

            state_payload = {
                "timestamp": now_utc.isoformat(),
                "et": now_et.isoformat(),
                "stage": stage.name if stage else "legacy",
                "hourly_target_hour": target_hour,
                "close_state": close_info["state"],
                "close_go": close_info["go"],
                "close_reasons": close_info["reasons"],
                "close_target_et": (
                    close_info["target_et"].isoformat() if close_info.get("target_et") else None
                ),
                "close_freshness_gate_et": (
                    close_info["freshness_strict_et"].isoformat()
                    if close_info.get("freshness_strict_et")
                    else None
                ),
                "freshness_threshold_ms": FRESHNESS_THRESHOLD_MS,
            }
            _write_state_artifact(state_payload, args.state_artifact)

            if args.once:
                break
            sleep_seconds = max(5.0, float(args.interval))
            time.sleep(sleep_seconds)
    except KeyboardInterrupt:
        print("[live-loop] received interrupt, exiting.", flush=True)


if __name__ == "__main__":
    main()
