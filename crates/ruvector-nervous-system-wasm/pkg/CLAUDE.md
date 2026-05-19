# ruvector-nervous-system-wasm/pkg

Generated `wasm-pack build` artifact tree, checked into the repo for use by
downstream JS consumers without needing the Rust toolchain.

## Files
- `package.json` - npm metadata (`ruvector-nervous-system-wasm`).
- `ruvector_nervous_system_wasm.js` - JS loader / bindings shim.
- `ruvector_nervous_system_wasm.d.ts` - TypeScript declarations for the
  JS shim.
- `ruvector_nervous_system_wasm_bg.wasm` - the compiled WASM module
  (size-optimized, target <100KB).
- `ruvector_nervous_system_wasm_bg.wasm.d.ts` - TS declarations describing
  raw WASM exports.

Regenerate with `wasm-pack build --target web` (or matching) from the crate
root after editing `src/`.
