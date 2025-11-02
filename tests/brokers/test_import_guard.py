from __future__ import annotations

from pathlib import Path

import kalshi_alpha.brokers.kalshi.live as live_broker


def test_live_broker_import_resolves_to_src() -> None:
    module_path = Path(live_broker.__file__ or "").resolve()
    path_str = module_path.as_posix()
    assert path_str.endswith("src/kalshi_alpha/brokers/kalshi/live.py")
    assert "exec/brokers" not in path_str
