# ruvector-delta-wasm/src

Source for the delta-operations WASM module. Each file is a private module
re-exported from `lib.rs`.

## Files
- `lib.rs` - `mod` declarations and glob re-exports of public types.
- `capture.rs` - `#[wasm_bindgen]` exports for capturing a delta from two
  vectors. Reports sparsity for downstream optimization.
- `apply.rs` - `#[wasm_bindgen]` exports for applying a captured delta to a
  base vector in place.
- `memory.rs` - shared-memory / typed-array helpers using `parking_lot::Mutex`
  and `smallvec` to minimize JS<->Rust copies.
- `simd.rs` - SIMD kernels (active when the `simd` cargo feature is on).
