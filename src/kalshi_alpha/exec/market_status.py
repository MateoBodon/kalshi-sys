"""Print Polygon/Massive market status for ops checks."""

from __future__ import annotations

import argparse
import json
from typing import Any

from kalshi_alpha.drivers.polygon_index.client import PolygonIndicesClient


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print Polygon market status (marketstatus/now).")
    parser.add_argument("--json", action="store_true", help="Emit raw JSON payload")
    args = parser.parse_args(list(argv) if argv is not None else None)

    client = PolygonIndicesClient()
    payload = client.fetch_market_status()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    server_time = payload.get("serverTime")
    market = payload.get("market")
    indices_groups = payload.get("indicesGroups")
    summary: dict[str, Any] = {
        "serverTime": server_time,
        "market": market,
    }
    if isinstance(indices_groups, dict):
        summary["indicesGroups"] = {
            key: indices_groups.get(key)
            for key in ("s_and_p", "nasdaq")
            if key in indices_groups
        }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
