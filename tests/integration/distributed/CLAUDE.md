Integration tests for ruvector's distributed-systems components: Raft consensus, multi-master replication, and auto-sharding. Run against a 5-node cluster spun up via docker-compose. Simulates an E2B-sandbox-like environment.

Files:
- `mod.rs` - declares the test modules (raft, replication, sharding, cluster_integration, performance_benchmarks).
- `raft_consensus_tests.rs` - leader election, log replication, safety invariants.
- `replication_tests.rs` - multi-master replication correctness.
- `sharding_tests.rs` - consistent-hashing-based auto-sharding.
- `cluster_integration_tests.rs` - end-to-end cluster scenarios (~12KB).
- `performance_benchmarks.rs` - latency/throughput under cluster load.
- `Dockerfile` - production-ish node image, built from repo root (`../../../`).
- `Dockerfile.test` - lighter image variant for test harnesses.
- `docker-compose.yml` - 5-node cluster topology (`raft-node-1`..`raft-node-5`) with `RAFT_PORT=7000`, `CLUSTER_PORT=8000`, `REPLICATION_PORT=9000`.
- `node_runner.sh` - container entrypoint that prints node config and runs a stub health server.

Bring up the cluster with `docker compose up -d` from this directory, then run the Rust tests against it. Related crates: `../../../crates/ruvector-raft/`, `../../../crates/ruvector-replication/`, `../../../crates/ruvector-cluster/`.
