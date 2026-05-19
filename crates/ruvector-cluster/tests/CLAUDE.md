# ruvector-cluster/tests

- `integration_tests.rs` — end-to-end cluster scenarios: node join/leave, gossip discovery convergence, consistent-hash rebalancing, consensus voting across simulated nodes.

Uses `tokio` multi-thread runtime (see dev-deps in `../Cargo.toml`). See `../CLAUDE.md`.
