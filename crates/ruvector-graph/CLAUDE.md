# ruvector-graph

Distributed Neo4j-compatible hypergraph database with SIMD optimization. Supports property graphs, hypergraphs (N-ary edges), Cypher queries, ACID transactions, vector-graph hybrid queries (RAG/GNN), and optional distributed sharding/federation/gossip via Raft + replication.

## Important files

- `Cargo.toml` — Depends on `ruvector-core` (simd+parallel), optional `ruvector-raft`/`ruvector-cluster`/`ruvector-replication`. Storage via `redb`, `memmap2`, `hnsw_rs`. SIMD via `simsimd`, `rayon`. Serialization via `rkyv`, `bincode`. Tokio optional for non-WASM async.
- `ARCHITECTURE.md` — Crate-level architecture overview.
- `src/lib.rs` — Crate root. Declares all top-level modules and re-exports `Edge`, `Hyperedge`, `Node`, `GraphDB`, `Transaction`, `EdgeId`, `NodeId`, etc.

## Source layout (`src/`)

Top-level files:
- `lib.rs` — Module declarations + re-exports.
- `graph.rs` — `GraphDB` primary entry type.
- `node.rs` / `edge.rs` / `hyperedge.rs` — Core graph entities (with builders).
- `transaction.rs` — `Transaction`, `TransactionManager`, `IsolationLevel`.
- `storage.rs` — `GraphStorage` (gated on `storage` feature).
- `index.rs` — Property/index abstractions.
- `error.rs` — `GraphError` + `Result`.
- `types.rs` — `NodeId`, `EdgeId`, `Label`, `Properties`, `PropertyValue`, `RelationType`.

Submodules (each with their own CLAUDE.md):
- `cypher/` — Lexer, parser, AST, semantic analysis, optimizer for Cypher.
- `executor/` — Query plan, operators, parallel pipeline, cache.
- `optimization/` — SIMD traversal, bloom filters, cache hierarchy, adaptive radix, JIT, memory pool, index compression.
- `hybrid/` — Vector-graph hybrid: semantic search, RAG, GNN, vector index, Cypher extensions.
- `distributed/` — `coordinator`, `federation`, `gossip`, `rpc`, `shard`, `replication` (feature-gated).

## Tests / Benches / Examples / Fuzz

- `tests/` — Compatibility, concurrency, Cypher exec + parser, edge, hyperedge, node, distributed, transaction, performance.
- `tests/fixtures/` — JSON datasets (`movie_database`, `social_network`, `expected_results`).
- `benches/` — Cypher parser, distributed query, graph + traversal, hybrid vector-graph, query execution, SIMD ops.
- `examples/test_cypher_parser.rs`.
- `fuzz/` — `cargo-fuzz` harness for Cypher parser.

## Performance targets

100x speedup over Neo4j via SIMD-vectorized traversal, cache-optimized layouts, bloom filters, ART, JIT, etc.

## Related

- Core: `ruvector-core`.
- Distributed: `ruvector-raft`, `ruvector-cluster`, `ruvector-replication`.
- PostgreSQL-side graph functions in `ruvector-postgres/src/graph/`.
