# RUN — TICKET-120A

Summary:
- Added a pilot readiness NO-GO `How to fix` section and sanitized ramp global reasons for portable markdown.
- Expanded readiness tests to cover NO-GO remediation and path sanitization.

Decisions:
- Sanitize global reasons at markdown render time to avoid mutating the policy JSON payload.

Risks:
- None observed; tests now pass in the local venv.
