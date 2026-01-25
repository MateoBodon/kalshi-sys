# PROJECT.md

## Project Profile
- Name: kalshi-sys
- One-liner: Python 3.11+ monorepo for Kalshi index ladder pricing, scanning, and paper-first execution.
- Type: trading
- Risk tier: high
- Primary languages: Python
- External dependencies / services: Kalshi API, Polygon.io, AWS/CloudWatch (optional)

## Goals (what "done" looks like)
- Maintain a paper-first, fail-closed index ladder pipeline for INX/INXU and NASDAQ100/NASDAQ100U.
- Provide deterministic ingestion, scanning, and reporting with offline fixtures.
- Keep ops tooling, runbooks, and readiness artifacts reproducible.

## Non-goals (explicitly out of scope)
- Live trading without explicit acknowledgement and human review.
- Expanding beyond index ladders or non-Polygon data vendors.
- Supporting non-index macro markets.

## Current state
- What works: index ladder ingestion, scans, reports, and safety gates (paper-first).
- What's missing: confirm readiness/ops gaps as needed.
- What's broken: unknown; needs verification.
- Biggest risks: accidental scope drift or live enablement without gates.

## Quickstart (how to run)
- `python -m venv .venv && . .venv/bin/activate`
- `pip install -e ".[dev]"`
- `pytest -q`

## Architecture (high-level)
- Modules: `src/kalshi_alpha`, `exec/`, `jobs/`, `monitor/`, `tools/`
- Data flow: Polygon/Kalshi -> ingest -> scan -> reports/artifacts.
- Key invariants: fail-closed safety gates; paper-only by default; maker-only enforcement in pilot/live.

## Constraints / preferences
- Performance constraints: keep scans deterministic and reproducible.
- Safety constraints: fail-closed, paper-only by default, explicit live ack.
- Style constraints: small diffs, tests for behavior changes.

## Links
- Docs: `README.md`, `docs/`
- Issues: (fill in)
