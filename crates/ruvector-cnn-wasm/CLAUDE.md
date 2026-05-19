# ruvector-cnn-wasm

WebAssembly bindings for `ruvector-cnn` — CNN-based image embedding extraction for the browser / Node. Re-exports a JS-friendly `WasmCnnEmbedder` plus contrastive-learning helpers.

## Features

- `default = ["console_error_panic_hook"]`
- `simd` — wires through to the SIMD-optimised convolutions in `ruvector-cnn`.

## Layout

- `Cargo.toml` — `cdylib` + `rlib`; path dep on `ruvector-cnn`; `wasm-bindgen`, `js-sys`, `serde-wasm-bindgen`. `[target.'cfg(target_arch = "wasm32")']` pulls in `getrandom` with the `js` feature.
- `src/lib.rs` — only source file.

## Public API (WASM)

- `init()` — `#[wasm_bindgen(start)]`, installs the panic hook.
- `EmbedderConfig { input_size, embedding_dim, normalize }` — JS-constructible config.
- `WasmCnnEmbedder` — `new(config)`, embed methods returning Float32Array.
- Wrapped contrastive losses: InfoNCE, triplet (`InfoNCELoss as RustInfoNCE`, `TripletLoss as RustTriplet`, `TripletDistance`).

## Related

- `crates/ruvector-cnn` — underlying Rust implementation (provides `contrastive` and `simd` modules).
- `crates/micro-hnsw-wasm`, `ruvector-mincut-gated-transformer-wasm`, `ruvector-verified-wasm`, `ruvector-learning-wasm` — sibling WASM modules.
