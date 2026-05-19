# ruvector-sparse-inference-wasm

WebAssembly bindings for PowerInfer-style sparse inference. Wraps `ruvector-sparse-inference` so GGUF models can run sparsely in the browser with optional streaming load and low-rank predictors.

## Important files

- `Cargo.toml` — `crate-type = ["cdylib", "rlib"]`. Depends on `ruvector-sparse-inference` (path), `wasm-bindgen`, `wasm-bindgen-futures`, `js-sys`, `web-sys` (Performance/Window/WorkerGlobalScope/Response), `getrandom`, `serde-wasm-bindgen`. Optional `console_error_panic_hook`. Release: `opt-level = 3`, LTO.
- `src/lib.rs` — Exports `SparseInferenceEngine` (constructor from GGUF bytes + JSON config, `load_streaming` from URL, `infer(input)`).

## Public API (JS)

- `class SparseInferenceEngine`:
  - `new(model_bytes: Uint8Array, config_json: string)`
  - `static async load_streaming(url: string, config_json: string)`
  - `infer(input: Float32Array) -> Float32Array`

## Tests

- `tests/web.rs` — `wasm-bindgen-test` browser tests.

## Related

- Backbone: `ruvector-sparse-inference` (model parsing, sparsity logic, predictors).
- Sibling WASM crates: `ruvector-attention-wasm`, `ruvector-tiny-dancer-wasm`, `ruvector-economy-wasm`.
