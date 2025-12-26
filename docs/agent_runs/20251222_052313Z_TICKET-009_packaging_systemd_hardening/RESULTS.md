# Results

- Changes:
  - Systemd unit now uses venv python with StartLimit settings in [Unit].
  - AWS/index ladder runbooks updated to use venv execution + EC2 bootstrap steps.
  - Added scipy/pandas runtime deps for index model imports.
- Bundle: docs/gpt_bundles/gpt_bundle_TICKET-009_packaging_systemd_hardening_20251222_052313Z_TICKET-009_packaging_systemd_hardening.zip
- Verifier: tools/verify_gpt_bundle.py (via make gpt-bundle)
