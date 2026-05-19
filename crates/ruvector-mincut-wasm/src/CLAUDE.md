# ruvector-mincut-wasm/src

Single-file wasm binding layer.

- `lib.rs` — defines `WasmMinCut`, `WasmThreeLevelHierarchy`, `WasmLocalKCut`, `WasmMinCutWrapper` and their JS-friendly methods. Installs `console_error_panic_hook` on init.

Build with `wasm-pack build --target web`.
