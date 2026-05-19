# ruvector-graph/benches

Criterion benchmarks for graph performance.

## Files

- `cypher_parser.rs` — Cypher tokenize+parse throughput.
- `query_execution.rs` — Full query execution latency.
- `graph_bench.rs` — Core graph CRUD.
- `graph_traversal.rs` — BFS/DFS traversal throughput.
- `hybrid_vector_graph.rs` — Hybrid vector+graph queries.
- `simd_operations.rs` — SIMD distance / predicate kernels.
- `distributed_query.rs` — Distributed query latency.
- `new_capabilities_bench.rs` — Benchmarks for recently added features.

Run via `cargo bench -p ruvector-graph`.
