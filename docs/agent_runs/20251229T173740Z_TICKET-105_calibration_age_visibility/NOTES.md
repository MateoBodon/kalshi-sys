# Notes

- `make calibrate-index` previously referenced `kalshi_alpha.jobs.*`, but the calibration modules live under the top-level `jobs/` package.
- The jobs modules import `kalshi_alpha`, so `PYTHONPATH=src` is required when running them via `python -m jobs.*`.
- `.env.local` is present; only env var names were recorded (no values).
