# edge-full/pkg/onnx

WASM build for ONNX-based embedding inference (HuggingFace-compatible) running directly in the browser.

## Files
- `ruvector_onnx_embeddings_wasm.js` - wasm-bindgen JS glue (ES module).
- `ruvector_onnx_embeddings_wasm.d.ts` - TypeScript declarations.
- `ruvector_onnx_embeddings_wasm_bg.js` - Auxiliary background glue.
- `ruvector_onnx_embeddings_wasm_bg.wasm` - Compiled WebAssembly module.
- `ruvector_onnx_embeddings_wasm_bg.wasm.d.ts` - Types for the raw WASM binding.

## Import path
`import { OnnxEmbedder } from '@ruvector/edge-full/onnx'`.

## Source
Built from the `ruvector-onnx-embeddings` Rust crate.
