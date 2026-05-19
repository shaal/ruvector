# ruvector-gnn-wasm/src

Single-file wasm binding layer.

- `lib.rs` — declares all `#[wasm_bindgen]` types and functions (query config, layer wrappers, tensor compression APIs, `differentiable_search`, `hierarchical_forward`). Init function installs `console_error_panic_hook` when enabled.

Build with `wasm-pack build --target web` (or `--target bundler`) — output is consumed by the JS workspace via `package.json`.
