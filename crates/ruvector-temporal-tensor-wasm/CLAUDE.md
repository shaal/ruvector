# ruvector-temporal-tensor-wasm

Thin WASM-target crate that re-exports the FFI surface of `ruvector-temporal-tensor` so it can be compiled to a `cdylib` for
WebAssembly consumers.

## Files

- `Cargo.toml` — `crate-type = ["cdylib"]`. Sole dep: `ruvector-temporal-tensor` with the `ffi` feature.
- `src/lib.rs` — single line: `pub use ruvector_temporal_tensor::ffi::*;`

## Related

- `../ruvector-temporal-tensor` — the underlying tensor compression implementation; this crate exists only to expose its `ffi`
  module to wasm32 targets.
- Other WASM siblings: `../ruvector-dag-wasm`, `../ruvector-math-wasm`.
