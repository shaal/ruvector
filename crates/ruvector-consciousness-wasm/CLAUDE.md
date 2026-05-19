# ruvector-consciousness-wasm

WASM bindings for `ruvector-consciousness`. Exposes JavaScript APIs for IIT Φ
(integrated information) computation, causal emergence analysis, and
quantum-inspired partition collapse.

## Important files
- `Cargo.toml` - `crate-type = ["cdylib", "rlib"]`. Pulls
  `ruvector-consciousness` with `wasm + phi + emergence + collapse`
  features. Release profile is size-optimized (`opt-level = "s"`, lto,
  panic=abort). Lints relaxed (research-tier crate).
- `src/lib.rs` - all `#[wasm_bindgen]` exports (single file).

## Public API surface
- `init()`, `version()` - module bootstrap.
- `WasmConsciousness` - main JS-facing engine; methods like `computePhi(...)`
  return JSON-serializable results.
- Wraps `ExactPhiEngine`, `SpectralPhiEngine`, `StochasticPhiEngine`,
  `GeoMipPhiEngine` (Φ); `CausalEmergenceEngine`, `RsvdEmergenceEngine`
  (emergence); `QuantumCollapseEngine` (collapse).
- Uses `ComputeBudget` + `TransitionMatrix` from the parent crate.

## Tests
- `dev-dependencies = wasm-bindgen-test`. No `tests/` dir in this crate; the
  underlying `ruvector-consciousness` carries the unit tests.

## Related
- `../ruvector-consciousness` - the IIT/emergence/collapse library.
- Sister WASM bindings: `ruvector-nervous-system-wasm`, `ruvector-graph-wasm`.
