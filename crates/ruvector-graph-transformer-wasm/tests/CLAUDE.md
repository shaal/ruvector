# ruvector-graph-transformer-wasm/tests

WASM integration tests, gated on `target_arch = "wasm32"`.

## Files

- `web.rs` — uses `wasm_bindgen_test` with `run_in_browser` config. Run via `wasm-pack test --headless --chrome` (or firefox/safari).
