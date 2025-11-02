# Changelog

## 2025-11-02
- Hardened the Kalshi HTTP client with header-only RSA-PSS signing (no bearer tokens), exponential backoff, and structured logging.
- Refactored `LiveBroker` to rely on the header-signed client, enforce idempotency via locking, and guard against duplicate submissions.
- Added integration tests validating signature construction, retry behaviour, query exclusion, and millisecond timestamps; refreshed broker safety tests to use the new client abstraction.
- Documented the credential expectations and connectivity flow in `.env.example`, `README.md`, and `docs/RUNBOOK.md`.
- Introduced an authenticated websocket client with reconnect/backoff, plus a live smoke CLI path (`sanity_check --live-smoke`) that exercises read-only REST checks.
