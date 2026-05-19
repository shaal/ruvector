# @ruvector/cnn

SIMD-optimized CNN feature-extraction for image embeddings, compiled from Rust to WebAssembly. Provides `CnnEmbedder`, `InfoNCELoss` (SimCLR-style contrastive loss), and a SIMD ops helper. Works in both browsers and Node.js — `init()` dynamically picks the right load path. All WASM artifacts are shipped in this directory (not a placeholder package).

## Important files

- `package.json` — `@ruvector/cnn` v0.1.0. Main `index.js`, ESM `index.mjs`, types `index.d.ts`. Build script invokes `wasm-pack build` against `../../crates/ruvector-cnn-wasm` and copies the output here.
- `index.js` — JS façade. Exports `init()`, `CnnEmbedder`, `InfoNCELoss`, `SimdOps`. Lazy-loads `./ruvector_cnn_wasm.js` and the `.wasm` binary, choosing browser vs Node loader.
- `index.d.ts` — TypeScript types for `EmbedderConfig`, `CnnEmbedder` (extract / cosineSimilarity / embeddingDim), `InfoNCELoss`, etc.
- `ruvector_cnn_wasm.js`, `ruvector_cnn_wasm.d.ts`, `ruvector_cnn_wasm_bg.wasm`, `ruvector_cnn_wasm_bg.wasm.d.ts` — wasm-pack-generated artifacts.

## Exports

`init`, `CnnEmbedder` (constructor: `{ inputSize=224, embeddingDim=512, normalize=true }`, methods `extract(imageData, w, h) -> Float32Array`, `cosineSimilarity(a, b)`, `embeddingDim`), `InfoNCELoss`, `SimdOps`.

## Scripts

- `build` — `wasm-pack build ../../crates/ruvector-cnn-wasm --target web --out-dir pkg`, plus `postbuild` that copies `pkg/*` into the package root.
- `test` — `node test.js`.

## Related

- Rust source: `crates/ruvector-cnn-wasm`.
- Sibling WASM packages: `npm/packages/acorn-wasm`, `npm/packages/ospipe-wasm`, `npm/packages/rudag` (pkg/).
