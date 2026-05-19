# ruvector-graph/src/distributed

Distributed graph query capabilities — feature-gated (`distributed`). Provides sharding, cross-cluster federation, graph-aware replication, and gossip-based membership.

## Files

- `mod.rs` — Re-exports `Coordinator`, `QueryPlan`, `ShardCoordinator`, `ClusterRegistry`, `FederatedQuery`, `Federation`, `RemoteCluster`, `GossipConfig`, `GossipMembership`, `MembershipEvent`, `NodeHealth`.
- `coordinator.rs` — Distributed query coordinator + plan building.
- `shard.rs` — Sharding/partitioning strategies.
- `federation.rs` — Multi-cluster federation primitives.
- `replication.rs` — Graph-aware replication (extends `ruvector-replication`).
- `gossip.rs` — Gossip-based cluster membership + health monitoring.
- `rpc.rs` — High-performance gRPC communication layer.

## Related

- Backbones: `ruvector-raft`, `ruvector-cluster`, `ruvector-replication`.
