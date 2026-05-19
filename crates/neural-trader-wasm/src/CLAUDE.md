# neural-trader-wasm/src

Single-file WASM glue: maps the Rust neural-trader API to JavaScript via
`wasm-bindgen` / `serde-wasm-bindgen`.

## Files
- `lib.rs` - `init`, `version`, `healthCheck`, plus `#[wasm_bindgen]` wrappers
  around `MarketEvent`, gate types from `neural-trader-coherence`, and the
  reservoir/memory stores from `neural-trader-replay`. Hex helpers convert
  16-byte event IDs to JS-friendly strings.
