# ruvector-mincut-gated-transformer-wasm/src

- `lib.rs` — single source file. Defines `init()`, `WasmTransformer { inner: MincutGatedTransformer, logits_buffer: Vec<i32> }`, `WasmGatePacket`, `WasmInferResult`, and `#[wasm_bindgen]` impls translating to/from the underlying `ruvector_mincut_gated_transformer` types.

See `../CLAUDE.md`.
