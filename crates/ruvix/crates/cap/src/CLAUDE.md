# ruvix-cap/src

## Files

- `lib.rs` — crate root, re-exports `CapabilityManager` + `CapManagerConfig`.
- `manager.rs` — `CapabilityManager`: top-level facade.
- `table.rs` — capability table data structure.
- `derivation.rs` — derivation tree (capabilities derived with equal or fewer rights).
- `grant.rs` — capability granting (including GRANT_ONCE non-transitive delegation).
- `revoke.rs` — revocation that propagates through the derivation tree.
- `boot.rs` — boot-time root-capability provisioning.
- `optimized.rs` — performance-tuned variant of hot lookup paths.
- `security.rs` — security invariants and audit helpers.
- `audit.rs` — capability-event audit log.
- `error.rs` — capability error enum.
