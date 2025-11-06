# Close Backtest Metrics

- Samples: 492
- Window: 2024-11-01 → 2025-10-31
- Contracts per trade: 100
- Taker fee formula: ceil(0.035 × contracts × price × (1 - price) × 100)/100

| Series | N | Mean Abs Error | Mean CRPS | Mean Brier | Mean PIT | Mean Taker EV ($) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NASDAQ100 | 246 | 0.0000 | 5.8969 | 0.2500 | 0.5000 | 49.12 |
| INX | 246 | 0.0000 | 1.3240 | 0.2500 | 0.5000 | 49.12 |
| ALL | 492 | 0.0000 | 3.6104 | 0.2500 | 0.5000 | 49.12 |

## PIT Histogram (deciles)

| Bin | Count | Frequency |
| --- | ---: | ---: |
| 0.0–0.1 | 0 | 0.000 |
| 0.1–0.2 | 0 | 0.000 |
| 0.2–0.3 | 0 | 0.000 |
| 0.3–0.4 | 0 | 0.000 |
| 0.4–0.5 | 0 | 0.000 |
| 0.5–0.6 | 492 | 1.000 |
| 0.6–0.7 | 0 | 0.000 |
| 0.7–0.8 | 0 | 0.000 |
| 0.8–0.9 | 0 | 0.000 |
| 0.9–1.0 | 0 | 0.000 |
