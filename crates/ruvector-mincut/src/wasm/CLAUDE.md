# ruvector-mincut/src/wasm

Wasm-friendly internal helpers compiled when the `wasm` feature is enabled (distinct from the standalone `crates/ruvector-mincut-wasm` crate, which provides the `#[wasm_bindgen]` surface).

## Files

- `mod.rs` — module wiring.
- `agentic.rs` — agentic / agent-oriented wasm helpers.
- `canonical.rs` — wasm helpers around the canonical decomposition.
- `simd.rs` — wasm SIMD path.
