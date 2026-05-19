# rvm-types/src

- `lib.rs` — crate root; declares and re-exports every module below.
- `ids.rs` — newtype IDs (`PartitionId`, etc.) — all `Copy + Clone + Eq + Hash`.
- `addr.rs` — address types (physical / virtual / guest physical).
- `capability.rs` — `Capability` token + rights bitmap.
- `witness.rs` — `WitnessRecord` (64 bytes, cache-line aligned).
- `memory.rs` — `MemoryRegion`, tier enum, mappings.
- `partition.rs` — partition-side types.
- `coherence.rs` — `CoherenceScore`, `CutPressure`.
- `device.rs` — `DeviceLease`.
- `scheduler.rs` — deadline / priority / mode types.
- `proof.rs` — `ProofTier`, hash types.
- `recovery.rs` — `RecoveryCheckpoint`.
- `config.rs` — compile-time configuration constants.
- `error.rs` — `RvmError`, `RvmResult`.

See `../CLAUDE.md`.
