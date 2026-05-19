# ui/ruvocal/static/wasm/

Compiled WebAssembly artifacts for the `rvagent_wasm` module, served at `/wasm/...` and consumed by `src/lib/wasm/`.

## Files

- `rvagent_wasm.js` — wasm-bindgen JS glue (loader / imports / exports surface).
- `rvagent_wasm_bg.wasm` — the WebAssembly binary.

These are build outputs from a Rust crate elsewhere in the monorepo (see `crates/` at the repo root). Do not edit by hand; replace by re-building from source.
