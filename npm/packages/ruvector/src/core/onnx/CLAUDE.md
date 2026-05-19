# src/core/onnx/

ONNX embeddings loader for `ruvector`.

- `loader.js` — runtime loader that instantiates the ONNX embeddings WASM module and exposes a uniform interface to `../onnx-embedder.ts` / `../onnx-optimized.ts`.
- `pkg/` — wasm-bindgen output containing the actual `.wasm` (MiniLM ONNX embeddings) and JS/d.ts shims.

Copied into `dist/core/onnx/pkg/` by the parent `build` script (`tsc && mkdir -p dist/core/onnx/pkg && cp src/core/onnx/pkg/package.json dist/core/onnx/pkg/`).
