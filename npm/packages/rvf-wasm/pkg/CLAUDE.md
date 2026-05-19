# pkg/

wasm-bindgen output for the RVF WASM microkernel (`crates/rvf/rvf-wasm`).

- `rvf_wasm.mjs` — ESM entry (parent package `main`).
- `rvf_wasm.js` — CJS-style glue.
- `rvf_wasm.d.ts` — TypeScript declarations.
- `rvf_wasm_bg.wasm` — compiled WASM (~42 KB), produced by `npm run build` and shrunk with `wasm-opt -Oz`.

Pure build output.
