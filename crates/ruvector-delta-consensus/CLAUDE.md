# ruvector-delta-consensus

Distributed delta consensus using CRDTs and causal ordering — enables consistent delta application across distributed nodes with vector-clock causality, conflict resolution, and delta compression for network transfer.

## Important files

- `Cargo.toml` — Depends on `ruvector-delta-core` (path, `serde` feature), `dashmap`, `parking_lot`, `smallvec`, `bincode 2.0-rc`, `uuid`, `chrono`. Optional `tokio` via `async` feature. Dev: `criterion`, `proptest`.
- `src/lib.rs` — Crate root. Declares modules and re-exports `CausalOrder`, `HybridLogicalClock`, `VectorClock`, `ConflictResolver`, `ConflictStrategy`, `MergeResult`, `DeltaCrdt`, `GCounter`, `LWWRegister`, `ORSet`, `PNCounter`. Defines `CausalDelta` and `ReplicaId`.

## Source modules (`src/`)

- `causal.rs` — `VectorClock`, `HybridLogicalClock`, `CausalOrder`.
- `crdt.rs` — Convergent replicated data types: `GCounter`, `PNCounter`, `LWWRegister`, `ORSet`, `DeltaCrdt` trait.
- `conflict.rs` — `ConflictResolver`, `ConflictStrategy`, `MergeResult`.
- `error.rs` — `ConsensusError` + `Result`.

## Public API

- `CausalDelta` (vector-clocked delta wrapper).
- CRDT types (`GCounter` etc.) and `DeltaCrdt` trait.
- `ConflictResolver` for merge strategies.

## Tests

- Uses `proptest` for property-based merge tests (inferred from dev-deps; benchmarks placeholder commented in `Cargo.toml`).

## Related

- Foundational: `ruvector-delta-core`.
- Other distributed primitives: `ruvector-cluster`, `ruvector-raft`, `ruvector-replication`.
