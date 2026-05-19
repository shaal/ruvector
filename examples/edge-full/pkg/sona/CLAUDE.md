# edge-full/pkg/sona

WASM build of SONA - the self-optimizing neural architecture with LoRA adapters - for the browser / edge.

## Files
- `ruvector_sona.js` - wasm-bindgen JS glue (ES module).
- `ruvector_sona.d.ts` - TypeScript declarations.
- `ruvector_sona_bg.wasm` - Compiled WebAssembly module.
- `ruvector_sona_bg.wasm.d.ts` - Types for the raw WASM binding.

## Import path
`import { SonaEngine } from '@ruvector/edge-full/sona'`.

## Source
Built from the `ruvector-sona` Rust crate.
