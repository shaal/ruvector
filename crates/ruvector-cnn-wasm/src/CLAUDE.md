# ruvector-cnn-wasm/src

- `lib.rs` — single source file. Defines `init`, `EmbedderConfig`, `WasmCnnEmbedder`, and thin wrappers around `ruvector_cnn::contrastive::{InfoNCELoss, TripletLoss, TripletDistance}` and `ruvector_cnn::simd`. All exported via `#[wasm_bindgen]`.

See `../CLAUDE.md`.
