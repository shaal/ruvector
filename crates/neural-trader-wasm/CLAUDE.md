# neural-trader-wasm

WASM bindings for the Neural Trader stack (ADR-085 / ADR-086). Exposes
JavaScript-friendly wrappers around `neural-trader-core` market events,
`neural-trader-coherence` gates, and `neural-trader-replay` reservoir memory.
`publish = false`.

## Important files
- `Cargo.toml` - `crate-type = ["cdylib", "rlib"]`. Workspace `wasm-bindgen` +
  `serde-wasm-bindgen`. Default feature toggles `console_error_panic_hook`.
- `Dockerfile.test` - container image used to run `wasm-pack test` in CI.
- `src/lib.rs` - all `#[wasm_bindgen]` exports.
- `tests/node-smoke.mjs` - Node-side smoke harness that loads the built
  package and exercises the JS bindings.

## Public API surface
- `init()` (#[wasm_bindgen(start)]) - sets the panic hook.
- `version()` -> String, `healthCheck()` -> bool.
- Re-exports / wrappers for: `MarketEvent`, `EventType`, `Side`,
  `CoherenceDecision`, `CoherenceGate`, `GateConfig`, `GateContext`,
  `RegimeLabel`, `ThresholdGate`, `MemoryStore`, `ReplaySegment`,
  `ReservoirStore`.
- Hex helpers for [u8; 16] event IDs.

## Build / test
- `wasm-pack build` (per workspace conventions; `wasm-opt = false` in
  release).
- `node tests/node-smoke.mjs` for the JS smoke test.

## Related
- `neural-trader-core`, `neural-trader-coherence`, `neural-trader-replay`,
  `neural-trader-strategies`.
