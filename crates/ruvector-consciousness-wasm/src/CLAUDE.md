# ruvector-consciousness-wasm/src

Single-file WASM glue for the consciousness library.

## Files
- `lib.rs` - `init`, `version`, and `WasmConsciousness` (the main JS handle).
  Wraps `PhiEngine` (Exact/Spectral/Stochastic/GeoMip), `EmergenceEngine`
  (Causal/Rsvd), and `ConsciousnessCollapse` (Quantum). Serializes results
  via `serde` + `serde-wasm-bindgen`.
