# ruvector-delta-wasm

WASM bindings for delta operations on vectors. Wraps `ruvector-delta-core` with
JS-friendly APIs for capturing, applying, and streaming vector deltas, with
optional SIMD and parallelism.

## Important files
- `Cargo.toml` - `crate-type = ["cdylib", "rlib"]`. Features:
  `simd` (forwards to `ruvector-delta-core/simd`), `parallel` (enables
  `rayon`), `console_error_panic_hook` (default).
- `src/lib.rs` - module wiring + re-exports.
- `src/capture.rs` - capture deltas from `(old, new)` vector pairs.
- `src/apply.rs` - apply deltas to a base vector.
- `src/memory.rs` - shared/typed-array memory plumbing for zero-copy from JS.
- `src/simd.rs` - SIMD-accelerated kernels.

## Public API surface
- `DeltaEngine` (JS class) - constructed with vector dim; `capture(old, new)`
  -> delta with sparsity metadata; `apply(base, delta)` mutates in place.
- Free function `vectorDelta(...)`.
- Wraps Rust `Delta` / engine types from `ruvector-delta-core`.

## Tests
- `dev-dependencies = wasm-bindgen-test`. No `tests/` dir present; tested
  via the upstream `ruvector-delta-core`.

## Related
- `../ruvector-delta-core` - native Rust delta library.
- Companion wasm crates: `ruvector-wasm`, `ruvector-graph-wasm`,
  `ruvector-solver-wasm`.
