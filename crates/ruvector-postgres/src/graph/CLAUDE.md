# ruvector-postgres/src/graph

Graph operations module — storage, traversal, Cypher, and W3C SPARQL 1.1 support persisted in PostgreSQL tables for durability across connections.

## Files

- `mod.rs` — Module entry; re-exports `cypher::execute_cypher`, `CypherQuery`, `storage::{Edge, EdgeStore, GraphStore, Node, NodeStore}`, `traversal::{bfs, dfs, shortest_path_dijkstra, PathResult}`.
- `storage.rs` — `Node`, `Edge`, `NodeStore`, `EdgeStore`, `GraphStore` backed by Postgres tables.
- `traversal.rs` — BFS, DFS, Dijkstra; `PathResult`.
- `operators.rs` — pgrx SQL operator wrappers.
- `cypher/` — Subset Cypher parser + executor (`parse_cypher`, `execute_cypher`).
- `sparql/` — Full SPARQL 1.1 implementation (parser/executor/functions/triple-store/results/AST).

## Pointers

- Pure-Rust standalone graph DB lives in the sibling crate `ruvector-graph` (more featureful).
- See `../../docs/GRAPH_IMPLEMENTATION.md` and `../../docs/GRAPH_QUICK_REFERENCE.md`.
