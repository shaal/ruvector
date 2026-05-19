# edge-full/pkg/edge

WASM build of the RuVector "edge" core: cryptographic identity, P2P, vector search (HNSW), and small neural networks for browsers.

## Files
- `ruvector_edge.js` - wasm-bindgen JS glue (ES module).
- `ruvector_edge.d.ts` - TypeScript declarations.
- `ruvector_edge_bg.wasm` - Compiled WebAssembly module.
- `ruvector_edge_bg.wasm.d.ts` - Types for the raw WASM binding.

## Import path
`import { WasmIdentity, WasmHnswIndex } from '@ruvector/edge-full/edge'`.

## Source
Built from the `ruvector-edge` Rust crate.
