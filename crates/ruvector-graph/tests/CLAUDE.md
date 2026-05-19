# ruvector-graph/tests

Integration tests for the graph database.

## Files

- `compatibility_tests.rs` — Neo4j-compatibility surface.
- `concurrent_tests.rs` — Concurrency / multi-threaded safety.
- `cypher_execution_tests.rs` — End-to-end Cypher execution.
- `cypher_parser_tests.rs` / `cypher_parser_integration.rs` — Parser correctness + integration with execution.
- `edge_tests.rs` / `node_tests.rs` / `hyperedge_tests.rs` — CRUD semantics for graph entities.
- `transaction_tests.rs` — ACID transaction semantics.
- `distributed_tests.rs` — Distributed coordinator/federation/gossip.
- `performance_tests.rs` — Perf regression guard.
- `fixtures/` — Shared JSON datasets.

Run via `cargo test -p ruvector-graph`.
