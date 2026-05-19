# OSpipe / src / wasm

`wasm-bindgen` surface that exposes a subset of OSpipe to JavaScript (browsers, Node, etc.). Compiled when targeting `wasm32-unknown-unknown`; native dependencies are cfg-gated out in `../../Cargo.toml`.

## Important files
- `mod.rs` - module root, only compiled on `wasm32`.
- `bindings.rs` - `#[wasm_bindgen]` exports (constructors, search/ingest methods) consumed by JS.
- `helpers.rs` - JS<->Rust conversion helpers (using `serde-wasm-bindgen`, `js-sys`).

## Build
- `wasm-pack build --target web ../../` (the crate's `crate-type` is `cdylib, rlib`).
- Packaged tarball: `../../dist/npm/ruvector-ospipe-wasm-0.1.0.tgz`.

## Related
- Native counterpart: `../server/` + `../bin/ospipe-server.rs`.
