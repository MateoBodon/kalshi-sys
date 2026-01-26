# NOTES

- Located gpt-bundle implementation in `Makefile` and validation in `tools/verify_gpt_bundle.py`.
- Baseline reproduction (dummy artifacts + bundle) already included fillcalib/pilot readiness/calibration entries; missing artifacts could not be reproduced locally.
- Added a Python staging helper to centralize bundle selection and enforce ARTIFACTS.md fail-closed checks, then wired Makefile to call it.
