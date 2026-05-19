# @ruvector/rvf-wasm

RuVector Format (RVF) WASM microkernel for browser and edge vector operations. Pure wasm-bindgen output; no TS layer.

## Key files

- `package.json` — `@ruvector/rvf-wasm` v0.1.6; ESM (`"type": "module"`); main `pkg/rvf_wasm.mjs`.
- `pkg/` — wasm-bindgen output.

## Published API / exports

- `.` -> `pkg/rvf_wasm.mjs` (types `pkg/rvf_wasm.d.ts`).
- `./wasm` -> raw `pkg/rvf_wasm_bg.wasm`.

## Scripts

- `build` -> `cargo build --release --target wasm32-unknown-unknown --manifest-path ../../crates/rvf/rvf-wasm/Cargo.toml && wasm-opt -Oz ... -o pkg/rvf_wasm_bg.wasm`

## Related

- Rust crate: `crates/rvf/rvf-wasm/`.
- Sibling: `npm/packages/rvf-solver/` (TS solver on top), `npm/packages/rvf-node/` (NAPI variant).
