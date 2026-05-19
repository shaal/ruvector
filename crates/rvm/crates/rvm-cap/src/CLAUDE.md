# rvm-cap/src

- `lib.rs` — crate root; declares modules and the public API.
- `table.rs` — capability table storage (fixed-size, no_std).
- `manager.rs` — `CapManager`: top-level grant / revoke / derive / verify entry points.
- `derivation.rs` — derivation tree (max depth 8).
- `grant.rs` — grant semantics, including `GRANT_ONCE` for non-transitive delegation.
- `revoke.rs` — epoch-based revocation propagating through the derivation tree.
- `verify.rs` — P1 capability check (< 1 us budget).
- `error.rs` — `CapError`.

See `../CLAUDE.md`.
