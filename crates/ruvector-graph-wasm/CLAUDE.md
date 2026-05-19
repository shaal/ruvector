# ruvector-graph-wasm

WebAssembly bindings for the RuVector graph database. Provides a
Neo4j-inspired API over `ruvector-graph`'s hypergraph infrastructure with
basic Cypher query support, async execution, and Web Workers.

## Important files
- `Cargo.toml` - `crate-type = ["cdylib", "rlib"]`. Pulls `ruvector-core`
  and `ruvector-graph` (with `wasm` feature). Uses `getrandom02 = "0.2"`
  aliased to get the `js` feature for WASM compatibility. `web-sys` is
  enabled with IndexedDB + Worker/MessagePort features.
- `build.sh` - convenience `wasm-pack build` script.
- `package.json` - npm metadata.
- `src/lib.rs` - `#[wasm_bindgen]` exports; module wiring (`async_ops`,
  `types`).
- `src/async_ops.rs` - Promise-returning async query and CRUD operations.
- `src/types.rs` - JS-friendly mirrors of graph types (`Node`, `Edge`,
  `Hyperedge`, `JsNode`, `JsEdge`, `JsHyperedge`, `QueryResult`,
  `GraphError`, helper `js_object_to_hashmap`).

## Public API surface
- `init()` (#[wasm_bindgen(start)]) - panic hook + `tracing-wasm`.
- Node / Edge / Hyperedge CRUD with hyperedge support for n-ary relationships.
- Basic Cypher queries (executed via `ruvector-graph`).
- Streaming async query execution.
- IndexedDB persistence (planned; web-sys bindings already present).
- Web Worker integration via `Worker` / `MessagePort` re-exports.

## Tests
None in this crate (no `tests/` dir). Tested upstream.

## Related
- `../ruvector-core` (hypergraph types), `../ruvector-graph` (database
  engine), `../ruvector-wasm` (vector DB WASM bindings).
- Consumed by `rvlite` (single-binary SQL/SPARQL/Cypher front-end).
