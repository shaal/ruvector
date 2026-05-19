# rvm-cap

Capability system for the RVM microhypervisor implementing the three-layer proof model from ADR-135.

| Layer | Name | Budget | v1 |
|-------|------|--------|----|
| P1 | Capability check | < 1 us | ship |
| P2 | Policy validation | < 100 us | ship |
| P3 | Deep proof | < 10 ms | deferred |

## Key concepts

- Unforgeable kernel-managed capability tokens with rights bitmaps.
- Parent/child derivation tree with monotonic right attenuation.
- Max delegation depth = 8.
- Epoch-based revocation propagates through the tree.
- `GRANT_ONCE` for non-transitive delegation.

`#![no_std] #![forbid(unsafe_code)] #![deny(missing_docs)]`.

## Layout

- `Cargo.toml` — `rlib`; deps `rvm-types`, `spin`. Features `std`, `alloc` (forwarded to `rvm-types`).
- `src/lib.rs` — module wiring and public API.
- `src/table.rs` — capability table storage.
- `src/manager.rs` — `CapManager` orchestrating grant/revoke/derive/verify.
- `src/derivation.rs` — derivation-tree bookkeeping.
- `src/grant.rs` — granting / `GRANT_ONCE` semantics.
- `src/revoke.rs` — epoch-based revocation propagation.
- `src/verify.rs` — P1 capability checks.
- `src/error.rs` — `CapError`.

See `../CLAUDE.md`.
