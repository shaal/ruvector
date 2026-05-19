# onnx-embeddings-wasm/src

Rust source for the WASM embedding library.

## Files

- `lib.rs` - Public `wasm_bindgen` surface.
- `embedder.rs` - High-level `Embedder` exposed to JS.
- `model.rs` - ONNX model loader via `tract-onnx`.
- `tokenizer.rs` - HuggingFace tokenizers wrapper (WASM-compatible build).
- `pooling.rs` - Pooling strategies.
- `error.rs` - Error types.

## Related

- JS glue: `../loader.js`, `../parallel-*.mjs`.
- Native sibling: `examples/onnx-embeddings/src/`.
