# ruvector-graph-node/src

- `lib.rs` — NAPI entry point. Defines the `#[napi] GraphDatabase` struct that wraps `ruvector_core::advanced::hypergraph::{HypergraphIndex, CausalMemory}`, `ruvector_graph::{GraphDB, storage::GraphStorage, cypher::parse_cypher, node::NodeBuilder}`, and the in-crate `TransactionManager`.
- `types.rs` — JS-facing structs / enums (`JsNode`, `JsEdge`, `JsHyperedge`, `JsBatchInsert`, `JsBatchResult`, `JsGraphOptions`, `JsGraphStats`, `JsDistanceMetric`, `JsTemporalGranularity`, `JsQueryResult`, `JsDeleteResult`, `JsHyperedgeQuery`, `JsTemporalHyperedge`, `JsEdgeResult`, `JsNodeResult`, `JsHyperedgeResult`, `JsDeleteNodeOptions`, `JsDeleteNodeResult`).
- `transactions.rs` — `TransactionManager` plus JS transaction lifecycle (`begin`, `commit`, `rollback`).
- `streaming.rs` — async streaming query helpers exposed to JS.

See `../CLAUDE.md`.
