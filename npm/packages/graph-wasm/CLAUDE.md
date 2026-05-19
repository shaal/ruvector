# @ruvector/graph-wasm

Neo4j-compatible hypergraph database compiled to WebAssembly. Supports
Cypher queries, SIMD-accelerated traversals, knowledge graphs, and
embedding-aware hyperedges from both browser and Node.

## Important files
- `package.json` - npm metadata (`@ruvector/graph-wasm` v2.0.3, ESM
  `type: module`). Lists only the wasm artefacts as published files.
- `index.js` - Thin re-export of the generated wasm-bindgen bindings
  (`./ruvector_graph_wasm.js`), plus `default as init` for explicit
  WASM initialization.
- `index.d.ts` - Hand-curated TypeScript surface (`GraphDB`, `JsNode`,
  `JsEdge`, `JsHyperedge`, `QueryResult`, `GraphStats`, `version()`).

## Published assets (set by `files`)
- `ruvector_graph_wasm_bg.wasm` - The compiled binary.
- `ruvector_graph_wasm.js` / `ruvector_graph_wasm.d.ts` - wasm-bindgen
  generated glue + types.
- `README.md`.

## Build
No local scripts; artefacts are produced from
`../../../crates/ruvector-graph-wasm` via `wasm-pack` and copied here
before publish.

## Related
- Rust crate: `../../../crates/ruvector-graph-wasm` (package.json
  `repository.directory` points to this).
