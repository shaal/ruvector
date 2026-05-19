# replication/src

TypeScript source for `@ruvector/replication`. Compiled to `dist/` via `tsc`.

## Files

- `index.ts` — barrel export. Re-exports all public types and classes (`ReplicaSet`, `SyncManager`, `ReplicationLog`, `VectorClock`, `ReplicaRole`, `SyncMode`, `ChangeOperation`, etc.) with a JSDoc usage example.
- `types.ts` — shared type definitions: `ReplicaId`, `LogicalClock`, `ReplicaRole`, `ReplicaStatus`, `SyncMode`, `HealthStatus`, `Replica`, `ChangeOperation`, `ChangeEvent`, `VectorClockValue`, `ReplicaSetConfig`, `SyncConfig`, `LogEntry`, `ReplicationError`, `ReplicationEvent`, `FailoverPolicy`.
- `replica-set.ts` — `ReplicaSet` class (EventEmitter3) managing the set of replicas, their roles/status, and failover policy.
- `sync-manager.ts` — `SyncManager` orchestrating change recording and propagation across replicas in sync/async/semi-sync modes.
- `vector-clock.ts` — `VectorClock` ops + `VectorClockComparison` enum (Before / After / Concurrent) for causality detection.

Each `.ts` has compiled `.js`, `.d.ts`, and `.map` siblings.
