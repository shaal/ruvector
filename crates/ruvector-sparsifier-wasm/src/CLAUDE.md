# ruvector-sparsifier-wasm/src

Single-file WASM binding crate.

## Files

- `lib.rs` — wraps `ruvector_sparsifier::{traits::Sparsifier, AdaptiveGeoSpar, SparseGraph, SparsifierConfig}` and exposes `WasmSparseGraph`, `WasmSparsifier`, plus the `init()` / `version()` / `default_config()` free functions.
