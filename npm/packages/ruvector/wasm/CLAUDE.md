# wasm/

Prebuilt WASM module shipped with the `ruvector` package — exposes the decompiler engine via WebAssembly.

- `ruvector_decompiler_wasm_bg.wasm` — compiled WASM (~1.4 MB).
- `ruvector_decompiler_wasm.js` — wasm-bindgen JS glue.
- `ruvector_decompiler_wasm.d.ts`, `*_bg.wasm.d.ts` — TypeScript declarations.
- `package.json` — minimal manifest for the wasm subpackage.

Published via `files: ["wasm/"]`. Pure build output.
