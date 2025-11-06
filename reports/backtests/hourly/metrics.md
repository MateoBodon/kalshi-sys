# Hourly Backtest Metrics

- Samples: 2979
- Window: 2024-11-01 → 2025-10-31
- Contracts per trade: 100
- Taker fee formula: ceil(0.035 × contracts × price × (1 - price) × 100)/100

| Series | N | Mean Abs Error | Mean CRPS | Mean Brier | Mean PIT | Mean Taker EV ($) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NASDAQ100U | 1488 | 0.0080 | 3.3414 | 0.2501 | 0.4999 | 49.13 |
| INXU | 1491 | 0.0073 | 0.6948 | 0.2499 | 0.5001 | 49.11 |
| ALL | 2979 | 0.0076 | 2.0168 | 0.2500 | 0.5000 | 49.12 |

## PIT Histogram (deciles)

| Bin | Count | Frequency |
| --- | ---: | ---: |
| 0.0–0.1 | 0 | 0.000 |
| 0.1–0.2 | 0 | 0.000 |
| 0.2–0.3 | 0 | 0.000 |
| 0.3–0.4 | 1 | 0.000 |
| 0.4–0.5 | 1 | 0.000 |
| 0.5–0.6 | 2977 | 0.999 |
| 0.6–0.7 | 0 | 0.000 |
| 0.7–0.8 | 0 | 0.000 |
| 0.8–0.9 | 0 | 0.000 |
| 0.9–1.0 | 0 | 0.000 |
