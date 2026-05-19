# ruvector-graph-node

Node.js native addon (NAPI-RS) wrapping the RuVector graph database. Exposes a high-performance native graph DB with Cypher-like queries, hypergraph capabilities (`CausalMemory`, `HypergraphIndex`), async/await, and zero-copy buffer sharing.

## Layout

- `Cargo.toml` — `cdylib` only. Depends on `ruvector-core`, `ruvector-graph` (with `storage` feature), `napi`, `napi-derive`, `tokio`, `futures`, `uuid`.
- `build.rs` — invokes `napi-build`.
- `src/lib.rs` — entry point. Defines `GraphDatabase` (`#[napi]`) wrapping `HypergraphIndex`, `CausalMemory`, `TransactionManager`, `GraphDB`, optional `GraphStorage`.
- `src/types.rs` — JS-facing types (`JsNode`, `JsEdge`, `JsHyperedge`, `JsTemporalHyperedge`, `JsBatchInsert`, `JsBatchResult`, `JsGraphOptions`, `JsGraphStats`, `JsDistanceMetric`, `JsTemporalGranularity`, `JsQueryResult`, etc.).
- `src/transactions.rs` — `TransactionManager` + JS transaction wrappers.
- `src/streaming.rs` — streaming query / result APIs for Node.

## Public API (JS / `#[napi]`)

`new GraphDatabase({ distanceMetric, dimensions, ... })` plus async methods for node/edge/hyperedge CRUD, Cypher parsing (`ruvector_graph::cypher::parse_cypher`), batch insert, hypergraph queries, transactions, and streaming.

## Related

- `crates/ruvector-core` — `DistanceMetric`, `advanced::hypergraph::{CausalMemory, Hyperedge, HypergraphIndex}`.
- `crates/ruvector-graph` — Rust-side graph DB and Cypher parser.
- `crates/ruvector-solver-node`, `crates/ruvector-cluster` — sibling Node bindings.
- Published as an `@ruvector/...` npm package (no `package.json` checked in here; built via `napi build`).
