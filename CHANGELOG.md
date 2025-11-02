# Changelog

## 2025-11-02
- Added resilient Kalshi HTTP client with RSA-PSS signing, bearer token refresh, retries/backoff, and structured logging.
- Refactored `LiveBroker` to use the new client, enforce idempotency through the client, and guard against duplicate submissions with thread-safe tracking.
- Introduced integration-style tests covering token issuance, retry behaviour, and clock skew validation; refreshed broker safety tests to use the HTTP client abstraction.
- Documented the new credential requirements and connectivity flow in `.env.example`, `README.md`, and `docs/RUNBOOK.md`.
