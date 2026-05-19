# ruvector-cluster

Distributed clustering and sharding for ruvector. Provides cluster-node management and health monitoring, consistent-hash shard routing, a DAG-based consensus protocol, and dynamic node discovery (gossip + static).

Companion to the `@ruvector/cluster` npm package (`package.json`).

## Layout

- `Cargo.toml` — depends on `ruvector-core`, `tokio`, `dashmap`, `parking_lot`, `uuid`, `chrono`, `futures`, `bincode`, `async-trait`. Lints relaxed (research-tier crate).
- `package.json` — npm metadata for `@ruvector/cluster`; `scripts.build = cargo build --release`.
- `src/` — four files; see `src/CLAUDE.md`.
- `tests/integration_tests.rs` — end-to-end cluster tests.

## Public API / key types

From `lib.rs`:
- `DagConsensus` (consensus module).
- `DiscoveryService`, `GossipDiscovery`, `StaticDiscovery` (discovery module).
- `ConsistentHashRing`, `ShardRouter` (shard module).
- `ClusterError` (`NodeNotFound`, `ShardNotFound`, `ConsensusError`, `DiscoveryError`, `NetworkError`, ...), `Result<T>` alias.
- Cluster-node status types.

## Related

- `crates/ruvector-core` — shared core types (path dep).
- `crates/ruvector-dag/src/qudag/consensus.rs` — alternative quantum-resistant consensus path.
- `crates/ruvector-hailo-cluster` — specialised cluster coordinator for Hailo embedding workers.
- npm: `npm/cluster` (or wherever `@ruvector/cluster` is published from).
