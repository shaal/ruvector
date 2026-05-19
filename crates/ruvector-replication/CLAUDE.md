# ruvector-replication

Data replication and synchronization for ruvector. Provides multi-node replica management, sync/async/semi-sync replication modes,
conflict resolution via vector clocks and CRDTs, change-data-capture streaming, and automatic failover with split-brain prevention.

## Files

- `Cargo.toml` — depends on `ruvector-core`, tokio (`time`), serde, thiserror, tracing, dashmap, parking_lot, uuid, chrono,
  futures, rand, bincode. `README.md` referenced.
- `src/lib.rs` — public API and worked example showing `ReplicaSet`, `ReplicaRole`, `SyncMode`, `SyncManager`,
  `ReplicationLog`.

## Public API surface (re-exported from `lib.rs`)

- `conflict::{ConflictResolver, LastWriteWins, MergeFunction, VectorClock}`
- `failover::{FailoverManager, FailoverPolicy, HealthStatus}`
- `replica::{Replica, ReplicaRole, ReplicaSet, ReplicaStatus}`
- Likely also `stream` and `sync` types (see those modules).

## Related

- `../ruvector-core` — base store and types being replicated.
- `../ruvector-cluster`, `../ruvector-quorum` (if present) — companion distributed-systems crates.
