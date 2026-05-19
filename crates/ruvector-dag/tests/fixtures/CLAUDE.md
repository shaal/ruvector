# ruvector-dag/tests/fixtures

Shared helpers used across the integration test suite.

- `mod.rs` — re-exports the helpers below.
- `dag_generator.rs` — synthetic `QueryDag` builders.
- `mock_qudag.rs` — in-memory `QuDagClient` substitute for tests that should not touch the network.
- `pattern_generator.rs` — synthetic `DagPattern` for SONA reasoning-bank tests.
- `trajectory_generator.rs` — synthetic `DagTrajectory` traces.

See `../CLAUDE.md`.
