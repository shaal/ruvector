# ruvector-delta-consensus/src

Four flat modules implementing distributed CRDT delta consensus.

## Files

- `lib.rs` — Crate doc + module declarations + re-exports. Defines `ReplicaId = String` and `CausalDelta`.
- `causal.rs` — Vector clocks, hybrid logical clocks, `CausalOrder` ordering.
- `crdt.rs` — `DeltaCrdt` trait + concrete CRDTs (`GCounter`, `PNCounter`, `LWWRegister`, `ORSet`).
- `conflict.rs` — Conflict-resolution strategies and merge results.
- `error.rs` — `ConsensusError` enum (thiserror) and crate `Result`.
