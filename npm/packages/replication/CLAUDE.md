# @ruvector/replication

TypeScript data replication / synchronization library — multi-node replicas with primary/secondary roles, sync/async/semi-sync modes, vector-clock conflict resolution, change-data capture, and automatic failover. Pure-TS (no native bindings); intended as the JS-side mirror of `crates/ruvector-replication`.

## Important files

- `package.json` — `@ruvector/replication` v0.1.0. Main `dist/index.js`, types `dist/index.d.ts`. Exports `.` (dual import/require → same files). Dep: `eventemitter3`. Scripts: `build` (tsc), `test` (`node --test`), `typecheck`, `clean`.
- `src/index.ts` — barrel: re-exports types (`ReplicaId`, `LogicalClock`, `ReplicaRole`, `ReplicaStatus`, `SyncMode`, `HealthStatus`, `Replica`, `ChangeOperation`, `ChangeEvent`, `VectorClockValue`, ...) and classes (`ReplicaSet`, `SyncManager`, `ReplicationLog`, `VectorClock`, ...). Includes a usage example in the JSDoc header.
- `src/replica-set.ts` — `ReplicaSet` (EventEmitter3-based) that tracks replicas, roles, and status; raises `ReplicationEvent`s.
- `src/sync-manager.ts` — `SyncManager` that orchestrates sync/async/semi-sync replication and emits change events.
- `src/vector-clock.ts` — vector clock primitives and `VectorClockComparison` (Before / After / Concurrent).
- `src/types.ts` — shared type definitions and enums.

Compiled `.js`/`.d.ts`/`.map` artifacts are present beside each `.ts` source.

## Related

- Rust counterpart: `crates/ruvector-replication` (referenced in `homepage` URL).
- Used by the broader `ruvector` deployment stack.
