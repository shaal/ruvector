# ruvector-replication/src

Source modules implementing the replication subsystem.

## Files

- `lib.rs` — crate root + re-exports + worked example.
- `replica.rs` — `Replica`, `ReplicaRole` (Primary/Secondary), `ReplicaSet`, `ReplicaStatus` — node membership and roles.
- `sync.rs` — `SyncManager`, `SyncMode` (Sync / Async / SemiSync { min_replicas }), `ReplicationLog`.
- `conflict.rs` — `ConflictResolver`, `LastWriteWins`, `MergeFunction`, `VectorClock` — CRDT/vector-clock primitives.
- `failover.rs` — `FailoverManager`, `FailoverPolicy`, `HealthStatus` — automatic failover with split-brain prevention.
- `stream.rs` — change-data-capture (CDC) and streaming change feed.
