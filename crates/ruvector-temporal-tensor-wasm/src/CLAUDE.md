# ruvector-temporal-tensor-wasm/src

## Files

- `lib.rs` — one-liner: `pub use ruvector_temporal_tensor::ffi::*;`. All exposed symbols come from the FFI module of the parent
  crate; this file exists only to set up the `cdylib` for the wasm32 target.
