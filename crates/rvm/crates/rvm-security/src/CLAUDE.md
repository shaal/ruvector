# rvm-security/src

- `lib.rs` — crate root.
- `gate.rs` — unified security gate; single entry point that runs capability → proof → witness in sequence.
- `validation.rs` — input validation for security-critical parameters.
- `attestation.rs` — attestation chain construction and report generation.
- `budget.rs` — DMA and resource budget enforcement.

See `../CLAUDE.md`.
