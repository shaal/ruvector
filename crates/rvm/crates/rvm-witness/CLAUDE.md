# rvm-witness

Witness logging subsystem for the RVM microhypervisor (ADR-134). Implements 64-byte fixed witness records with an FNV-1a hash chain for a tamper-evident audit trail.

## Core invariant

**No witness, no mutation.** Every privileged action emits a witness record before the mutation commits. If emission fails, the mutation does not proceed.

## Record format (64 bytes, cache-line aligned)

| Offset | Size | Field |
|--------|------|-------|
| 0 | 8 | sequence (u64) |
| 8 | 8 | timestamp_ns (u64) |
| 16 | 1 | action_kind (u8) |
| 17 | 1 | proof_tier (u8) |
| 18 | 2 | flags (u16) |
| 20 | 4 | actor_partition_id (u32) |
| 24 | 4 | target_object_id (u32) |
| 28 | 4 | capability_hash (u32) |
| 32 | 8 | payload (u64) |
| 40 | 8 | prev_hash (u64) |
| 48 | 8 | record_hash (u64) |
| 56 | 8 | aux (u64) |

## Features

- `default = ["strict-signing", "crypto-sha256"]`.

## Layout

- `Cargo.toml` — `rlib`; deps `rvm-types`, `spin`; optional `hmac`, `sha2`.
- `src/lib.rs` — crate root.
- `src/record.rs` — fixed 64-byte `WitnessRecord` layout.
- `src/log.rs` — append-only witness log storage.
- `src/emit.rs` — emission API ("no witness, no mutation" invariant).
- `src/hash.rs` — FNV-1a hash-chain.
- `src/replay.rs` — log replay / verification.
- `src/signer.rs` — witness signing (`strict-signing`, `crypto-sha256` features).

See `../CLAUDE.md`.
