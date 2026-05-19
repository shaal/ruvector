# ruvector-nervous-system-wasm/tests

Browser-side integration tests using `wasm-bindgen-test`.

## Files
- `web.rs` - `#[wasm_bindgen_test]` tests for `BTSPLayer`, `Hypervector` /
  `HdcMemory`, `WTALayer` / `KWTALayer`, and `GlobalWorkspace`. Run with
  `wasm-pack test --headless --firefox` (or similar).
