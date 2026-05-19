# ruvector-cluster/src

- `lib.rs` — crate root, defines `ClusterError`, `Result`, node-status enum, and re-exports the sub-modules.
- `consensus.rs` — `DagConsensus` protocol (proposals, votes, finalisation over a DAG of states).
- `discovery.rs` — `DiscoveryService` trait plus `GossipDiscovery` and `StaticDiscovery` implementations.
- `shard.rs` — `ConsistentHashRing` and `ShardRouter` for distributing keys across nodes.

See `../CLAUDE.md`.
