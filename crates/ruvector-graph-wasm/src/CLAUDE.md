# ruvector-graph-wasm/src

WASM glue for the RuVector graph database.

## Files
- `lib.rs` - `init()` panic hook + module wiring; primary `#[wasm_bindgen]`
  surface. Wraps `ruvector_core::advanced::hypergraph::{Hyperedge,
  HypergraphIndex, TemporalHyperedge, TemporalGranularity}` and
  `DistanceMetric`. Uses `parking_lot::Mutex` for shared state.
- `async_ops.rs` - Promise-returning operations (async query, batch ops);
  bridges `wasm_bindgen_futures` to the synchronous graph engine.
- `types.rs` - JS-friendly type mirrors: `Node`, `Edge`, `Hyperedge` (Rust
  side) plus `JsNode`, `JsEdge`, `JsHyperedge` (JS-serializable),
  `QueryResult`, `GraphError`, `NodeId`/`EdgeId`/`HyperedgeId`. Helper
  `js_object_to_hashmap` converts JS objects to property maps.
