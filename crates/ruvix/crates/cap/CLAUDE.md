# ruvix-cap

seL4-inspired capability management for the RuVix Cognition Kernel (ADR-087 Section 6). All access control flows through
unforgeable capabilities; no syscall succeeds without an appropriate capability handle. Supports a derivation tree (revocation
propagates), GRANT_ONCE non-transitive delegation, epoch-based stale-handle invalidation, and a configurable max delegation depth
(default 8).

## Files

- `Cargo.toml` — depends on `ruvix-types`. Dev: criterion, proptest. `autobenches = false` (benches declared explicitly).
- `README.md` — public docs.
- `src/` — see `src/CLAUDE.md`.
- `benches/cap_bench.rs` — capability operation microbench.
- `tests/capability_test.rs`, `tests/security_test.rs` — functional + security tests.
