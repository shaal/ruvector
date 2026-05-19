# edge-full/pkg/rvlite

WASM build of `rvlite`: a SQL / SPARQL / Cypher vector database in the browser.

## Files
- `rvlite.js` - wasm-bindgen JS glue (ES module).
- `rvlite.d.ts` - TypeScript declarations.
- `rvlite_bg.wasm` - Compiled WebAssembly module.
- `rvlite_bg.wasm.d.ts` - Types for the raw WASM binding.

## Import path
`import { Database } from '@ruvector/edge-full/rvlite'`.

## Source
Built from the `rvlite` Rust crate.
