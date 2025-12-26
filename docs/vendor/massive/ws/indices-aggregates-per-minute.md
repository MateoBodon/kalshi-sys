# WEBSOCKET
## Indices

### Aggregates (Per Minute)

**Endpoint:** `WS /indices/AM`

**Description:**

Stream minute-by-minute aggregated OHLC (Open, High, Low, Close) for a specified index via WebSocket. These aggregates update continuously in Eastern Time (ET) and capture changes in the index’s values. Unlike stocks or options, index aggregates are derived from index values rather than individual trades. If no new index updates occur within a given minute, no bar is emitted. By providing an ongoing feed of updated market snapshots, this endpoint enables users to track intraday index movements, refine analysis, and power real-time market visualizations.

Use Cases: Real-time monitoring, dynamic charting, intraday trend analysis, market research.

## Query Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| `ticker` | string | Yes | Specify an index ticker using "I:" prefix or use * to subscribe to all index tickers. You can also use a comma separated list to subscribe to multiple index tickers. You can retrieve available index tickers from our [Index Tickers API](https://massive.com/docs/rest/indices/tickers/all-tickers).  |

## Sample Response

```json
{
  "ev": "AM",
  "sym": "I:SPX",
  "op": 3985.67,
  "o": 3985.67,
  "c": 3985.67,
  "h": 3985.67,
  "l": 3985.67,
  "s": 1678220675805,
  "e": 1678220675805
}
```
