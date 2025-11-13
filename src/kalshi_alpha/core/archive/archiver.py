"""Archive Kalshi public data snapshots for later replay."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path

from kalshi_alpha.core.kalshi_api import Event, KalshiPublicClient, Market, Orderbook, Series
from kalshi_alpha.datastore.paths import RAW_ROOT


def archive_scan(  # noqa: PLR0913
    series: Series | str,
    client: KalshiPublicClient,
    events: Iterable[Event],
    markets: Iterable[Market],
    orderbooks: Mapping[str, Orderbook],
    out_dir: Path | None = None,
    *,
    timestamp: datetime | None = None,
) -> Path:
    """Persist a snapshot of series/events/markets/orderbooks for later replay."""

    base_dir = Path(out_dir) if out_dir is not None else RAW_ROOT / "kalshi"
    generated_at = timestamp.astimezone(UTC) if timestamp is not None else datetime.now(tz=UTC)
    date_dir = base_dir / generated_at.date().isoformat()
    time_dir = date_dir / generated_at.strftime("%H%M%S")

    series_obj = series if isinstance(series, Series) else Series(id=str(series), ticker=str(series), name=str(series))
    series_slug = series_obj.ticker.upper()
    target_dir = time_dir / series_slug
    orderbooks_dir = target_dir / "orderbooks"
    orderbooks_dir.mkdir(parents=True, exist_ok=True)

    series_path = target_dir / "series.json"
    events_path = target_dir / "events.json"
    markets_path = target_dir / "markets.json"

    series_path.write_text(_dump_json(series_obj), encoding="utf-8")
    events_path.write_text(_dump_json(list(events)), encoding="utf-8")
    markets_path.write_text(_dump_json(list(markets)), encoding="utf-8")

    orderbook_entries: list[str] = []
    for market_id, orderbook in sorted(orderbooks.items()):
        ob_path = orderbooks_dir / f"{market_id}.json"
        ob_path.write_text(_dump_json(orderbook), encoding="utf-8")
        orderbook_entries.append(str(ob_path.relative_to(target_dir)))

    manifest = {
        "series": {
            "id": series_obj.id,
            "ticker": series_obj.ticker,
            "name": series_obj.name,
        },
        "generated_at": generated_at.isoformat(),
        "client": {
            "base_url": getattr(client, "base_url", None),
            "offline": getattr(client, "use_offline", False),
        },
        "paths": {
            "series": str(series_path.relative_to(target_dir)),
            "events": str(events_path.relative_to(target_dir)),
            "markets": str(markets_path.relative_to(target_dir)),
            "orderbooks": orderbook_entries,
        },
    }

    manifest_path = target_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def _dump_json(obj: object) -> str:
    return json.dumps(_to_jsonable(obj), indent=2, sort_keys=True)


def _to_jsonable(obj: object) -> object:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if is_dataclass(obj):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(item) for item in obj]
    if isinstance(obj, dict):
        return {str(key): _to_jsonable(value) for key, value in obj.items()}
    return obj
