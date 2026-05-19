# edge-full/pkg/graph

WASM build of the RuVector graph database with Cypher-style queries (Neo4j-style API in the browser).

## Files
- `ruvector_graph_wasm.js` - wasm-bindgen JS glue (ES module).
- `ruvector_graph_wasm.d.ts` - TypeScript declarations.
- `ruvector_graph_wasm_bg.wasm` - Compiled WebAssembly module.
- `ruvector_graph_wasm_bg.wasm.d.ts` - Types for the raw WASM binding.

## Import path
`import { WasmGraphStore } from '@ruvector/edge-full/graph'`.

## Source
Built from the `ruvector-graph` Rust crate.
