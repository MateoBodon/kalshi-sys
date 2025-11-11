from __future__ import annotations

import json
from pathlib import Path

import pytest

from kalshi_alpha.exec.monitors import fee_rules
from monitor import fee_rules_watch


def test_fee_rules_watch_ack_flow(tmp_path: Path, requests_mock) -> None:
    requests_mock.get("https://example.com/fees", text="fee v1")
    config = tmp_path / "config.yaml"
    config.write_text(
        """
        targets:
          - id: fees
            url: https://example.com/fees
        """.strip(),
        encoding="utf-8",
    )
    state = tmp_path / "state.json"
    artifact = tmp_path / "artifact.json"

    args = [
        "--config",
        str(config),
        "--state",
        str(state),
        "--artifact",
        str(artifact),
    ]
    rc = fee_rules_watch.main(args)
    assert rc == 1
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["status"] == "ALERT"
    assert payload["requires_ack"] is True

    requests_mock.get("https://example.com/fees", text="fee v2")
    rc = fee_rules_watch.main([*args, "--ack"])
    assert rc == 0
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["status"] == "OK"
    ready, reason = fee_rules.is_ready(payload)
    assert ready, f"fee rules not ready: {reason}"
