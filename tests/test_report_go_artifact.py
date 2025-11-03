from __future__ import annotations

import json
from pathlib import Path

from kalshi_alpha.exec.reports import write_markdown_report
from kalshi_alpha.exec.runners.scan_ladders import Proposal


def _sample_proposals() -> list[Proposal]:
    return [
        Proposal(
            market_id="M1",
            market_ticker="TENY_X",
            strike=100.0,
            side="YES",
            contracts=1,
            maker_ev=1.0,
            taker_ev=0.0,
            maker_ev_per_contract=0.5,
            taker_ev_per_contract=0.0,
            strategy_probability=0.55,
            market_yes_price=0.5,
            survival_market=0.45,
            survival_strategy=0.55,
            max_loss=0.5,
            strategy="TENY",
            metadata=None,
        )
    ]


def test_report_uses_artifact_go_status(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "reports" / "_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_dir / "go_no_go.json"
    artifact_path.write_text(json.dumps({"go": False, "reasons": ["test"]}), encoding="utf-8")

    report_dir = tmp_path / "reports" / "TENY"
    report_path = write_markdown_report(
        series="TENY",
        proposals=_sample_proposals(),
        ledger=None,
        output_dir=report_dir,
        monitors={},
        exposure_summary={},
        manifest_path=None,
        go_status=True,
        go_artifact_path=artifact_path,
    )

    first_line = report_path.read_text(encoding="utf-8").splitlines()[0]
    assert "**GO/NO-GO:** NO-GO" in first_line
