# ruvector-graph/src

Source root for the distributed Neo4j-compatible hypergraph database.

## Top-level files

- `lib.rs` — Module declarations + crate-wide re-exports.
- `graph.rs` — `GraphDB` (primary handle to a graph database).
- `node.rs` — `Node`, `NodeBuilder`.
- `edge.rs` — `Edge`, `EdgeBuilder`.
- `hyperedge.rs` — `Hyperedge`, `HyperedgeBuilder`, `HyperedgeId` (N-ary relationships).
- `transaction.rs` — `Transaction`, `TransactionManager`, `IsolationLevel`.
- `storage.rs` — `GraphStorage` (gated on `storage` feature).
- `index.rs` — Property index abstractions.
- `error.rs` — `GraphError` + `Result`.
- `types.rs` — `NodeId`, `EdgeId`, `Label`, `Properties`, `PropertyValue`, `RelationType`.

## Submodules

- `cypher/` — Cypher query language (lexer, parser, AST, semantic analysis, optimizer).
- `executor/` — Query execution engine (plan, operators, pipeline, cache, parallel).
- `optimization/` — Performance modules (SIMD traversal, bloom filter, ART, JIT, memory pool, cache hierarchy, index compression).
- `hybrid/` — Vector-graph hybrid (semantic search, RAG, GNN, vector index, Cypher extensions).
- `distributed/` — Feature-gated sharding, federation, gossip, RPC, replication.
