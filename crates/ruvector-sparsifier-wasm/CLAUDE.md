# ruvector-sparsifier-wasm

WASM bindings for `ruvector-sparsifier` — dynamic spectral graph sparsification in the browser or any WASM runtime.

## Layout

- `Cargo.toml` — `crate-type = ["cdylib", "rlib"]`, deps: `ruvector-sparsifier` (default-features=false, with `wasm` feature), wasm-bindgen + futures + serde stack, `getrandom` (js), `console_error_panic_hook`. `wasm-opt = false`.
- `src/lib.rs` — sole source. Exposes `init()`, `version()`, `default_config()`, `WasmSparseGraph`, and `WasmSparsifier` over `AdaptiveGeoSpar`, `SparseGraph`, `SparsifierConfig` from the native crate.

## Public JS API surface

- `init()` — panic-hook setup.
- `version()` — returns the underlying crate version.
- `default_config()` — pretty-JSON of `SparsifierConfig::default()`.
- `WasmSparseGraph::new(n)` — build a graph with `n` vertices then add edges.
- `WasmSparsifier` — wraps `AdaptiveGeoSpar` and exposes its `Sparsifier`-trait methods.

## Tests

None inside this crate.

## Related crates

- `crates/ruvector-sparsifier` — the native sparsifier crate.
