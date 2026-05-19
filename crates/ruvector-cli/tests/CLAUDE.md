# ruvector-cli/tests

Integration tests against the `ruvector` and `ruvector-mcp` binaries.

- `cli_tests.rs` — end-to-end CLI command coverage.
- `mcp_tests.rs` — MCP JSON-RPC handler tests over the in-process transport.
- `hooks_tests.rs` — hooks engine (in-memory and, when feature-gated, Postgres).
- `gnn_performance_test.rs` — perf regression for GNN inference path.
