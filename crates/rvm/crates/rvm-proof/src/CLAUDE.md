# rvm-proof/src

- `lib.rs` — crate root.
- `context.rs` — proof context with builder pattern for P2 validation.
- `engine.rs` — unified proof engine; runs the P1 → P2 → witness pipeline.
- `policy.rs` — P2 policy rules, evaluated in constant time.
- `constant_time.rs` — constant-time comparison utilities.
- `signer.rs` — witness signing traits + impls (ADR-142).
- `tee.rs` — TEE attestation trait definitions (ADR-142).
- `tee_provider.rs` — software TEE quote provider (ADR-142 Phase 3).
- `tee_verifier.rs` — software TEE quote verifier (ADR-142 Phase 3).
- `tee_signer.rs` — TEE-backed witness signer pipeline.

See `../CLAUDE.md`.
