# @ruvector/ruqu-wasm

Run quantum simulations in the browser. WebAssembly bindings for the
RuQu quantum-circuit engine: up to 25-qubit state-vector simulation,
VQE, Grover, QAOA, and surface-code error correction.

## Important files
- `package.json` - npm metadata (`@ruvector/ruqu-wasm` v2.0.5, ESM
  `type: module`). Lists only the wasm artefacts as published files.
- (No `src/` here — the package directory just hosts the published
  wasm-bindgen output; build artefacts are copied from the Rust crate.)

## Published assets (set by `files`)
- `ruqu_wasm_bg.wasm` - Compiled quantum simulator.
- `ruqu_wasm.js` / `ruqu_wasm.d.ts` - wasm-bindgen glue and types.
- `README.md`.

## Build
Produced from `../../../crates/ruqu-wasm` via `wasm-pack`; outputs are
copied into this directory before publish (the package.json
`repository.directory` field points back at the crate).

## Related
- Rust crate: `../../../crates/ruqu-wasm`.
- Other RuQu crates: `../../../crates/{ruQu,ruqu-algorithms,ruqu-core,
  ruqu-exotic}`.
