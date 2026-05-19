# pkg/

wasm-bindgen output for the ONNX embeddings WASM module used by `ruvector`.

- `ruvector_onnx_embeddings_wasm_bg.wasm` — the compiled WASM (~7 MB) with MiniLM embeddings.
- `ruvector_onnx_embeddings_wasm.js` / `_bg.js` — wasm-bindgen JS glue.
- `ruvector_onnx_embeddings_wasm.d.ts`, `_bg.wasm.d.ts` — TypeScript declarations.
- `loader.js` — wrapper loader (duplicated from `../loader.js`).
- `package.json` — minimal manifest (copied to `dist/core/onnx/pkg/` by the parent build).
- `LICENSE`.

Pure build output; do not edit by hand.
