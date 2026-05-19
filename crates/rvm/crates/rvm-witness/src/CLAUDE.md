# rvm-witness/src

- `lib.rs` — crate root.
- `record.rs` — fixed 64-byte `WitnessRecord` (cache-line aligned).
- `log.rs` — append-only witness log storage.
- `emit.rs` — emission API enforcing the "no witness, no mutation" invariant.
- `hash.rs` — FNV-1a hash chain over the log.
- `replay.rs` — log replay / verification.
- `signer.rs` — witness signing (features `strict-signing`, `crypto-sha256`; optional `hmac` / `sha2`).

See `../CLAUDE.md`.
