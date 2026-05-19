# ruvector-attention-wasm/tests

WASM browser tests using `wasm-bindgen-test`.

## Files

- `web.rs` — Targets the `web` runner; exercises the attention mechanism wrappers end-to-end inside a headless browser.

Run via `wasm-pack test --headless --chrome crates/ruvector-attention-wasm`.
