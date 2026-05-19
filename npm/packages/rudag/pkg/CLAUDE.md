# rudag/pkg

`wasm-pack` output for the bundler target (browsers / bundlers). Generated from `crates/ruvector-dag-wasm` via `npm run build:wasm:bundler` and consumed by `src/dag.ts` and the `./wasm` subpath export.

## Files

- `package.json` — generated `ruvector-dag-wasm` package manifest (ESM `module`, `types`, `sideEffects`).
- `ruvector_dag_wasm.js`, `ruvector_dag_wasm_bg.js` — JS glue.
- `ruvector_dag_wasm.d.ts`, `ruvector_dag_wasm_bg.wasm.d.ts` — type declarations.
- `ruvector_dag_wasm_bg.wasm` — the WASM binary.

Do not edit by hand — regenerated on every build.
