Container for cross-crate integration test suites that require infrastructure beyond a single `cargo test` invocation.

Subdirectories:
- `distributed/` - 5-node Raft + replication + sharding cluster tests driven by docker-compose.

Add new subdirectories here for additional multi-process or multi-service integration scenarios.
