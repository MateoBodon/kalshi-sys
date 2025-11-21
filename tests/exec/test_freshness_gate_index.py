from kalshi_alpha.exec.runners import scan_ladders


def test_freshness_ignores_non_polygon_stale():
    summary = {
        "status": "OK",
        "required_feeds_ok": False,
        "stale_feeds": ["dol_claims.latest_report", "treasury_10y.daily"],
    }
    reason = scan_ladders._freshness_fatal_reason(summary, require_polygon_ws=True)
    assert reason is None


def test_freshness_flags_polygon_stale():
    summary = {
        "status": "OK",
        "required_feeds_ok": False,
        "stale_feeds": ["polygon_index.websocket"],
    }
    reason = scan_ladders._freshness_fatal_reason(summary, require_polygon_ws=True)
    assert reason == "polygon_ws_stale"
