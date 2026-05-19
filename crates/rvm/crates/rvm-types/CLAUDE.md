# rvm-types

Foundation types for the RVM coherence-native microhypervisor (ADR-132/133/134). Minimal external deps (only `bitflags`); provides the shared type vocabulary for every other RVM crate.

## First-class objects (ADR-132)

| Type | Purpose |
|------|---------|
| `PartitionId` | Coherence-domain container; unit of scheduling, isolation, migration |
| `Capability` | Unforgeable authority token; specific rights over specific objects |
| `WitnessRecord` | 64-byte audit record emitted by every privileged action |
| `MemoryRegion` | Typed, tiered, owned memory range with explicit lifetime |
| `CommEdge` | Inter-partition communication channel; weighted edge in the coherence graph |
| `DeviceLease` | Time-bounded, revocable access to a hardware device |
| `CoherenceScore` | Locality / coupling metric from the coherence graph |
| `CutPressure` | Graph-derived isolation signal; high pressure triggers migration / split |
| `RecoveryCheckpoint` | Snapshot for rollback and reconstruction |

`#![no_std] #![forbid(unsafe_code)] #![deny(missing_docs)]`. Zero heap allocation by default; all identifiers are `Copy + Clone + Eq + Hash`-compatible newtypes.

## Features

- `default = []`
- `std`, `alloc` (purely additive — no behaviour change for no_std consumers).

## Layout

- `Cargo.toml` — `rlib`; only dep is `bitflags`.
- `src/lib.rs` — crate root, declares + re-exports all type modules.
- `src/ids.rs` — newtype identifiers (`PartitionId`, ...).
- `src/addr.rs` — physical / virtual / guest physical address types.
- `src/capability.rs` — `Capability` token and rights bitmap.
- `src/witness.rs` — `WitnessRecord` (64-byte fixed layout).
- `src/memory.rs` — `MemoryRegion`, tier enum.
- `src/partition.rs` — partition-related types.
- `src/coherence.rs` — `CoherenceScore`, `CutPressure`.
- `src/device.rs` — `DeviceLease`.
- `src/scheduler.rs` — deadline / priority types.
- `src/proof.rs` — `ProofTier`, hashes.
- `src/recovery.rs` — `RecoveryCheckpoint`.
- `src/config.rs` — compile-time configuration constants.
- `src/error.rs` — `RvmError`, `RvmResult`.

See `../CLAUDE.md`.
