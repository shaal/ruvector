# onnx-embeddings/src

Library + CLI sources for `ruvector-onnx-embeddings`.

## Files

- `lib.rs` - Public API.
- `main.rs` (~10KB) - CLI entrypoint.
- `embedder.rs` - High-level `Embedder` API.
- `model.rs` - ONNX model loader (uses `ort`).
- `tokenizer.rs` - HuggingFace tokenizer wrapper.
- `pooling.rs` - Mean/CLS/max pooling strategies.
- `config.rs` - Configuration types and defaults.
- `error.rs` - Error enum.
- `ruvector_integration.rs` - Bridge to the `ruvector` vector store.
- `gpu/` - GPU acceleration backend (wgpu/CUDA/TensorRT/CoreML/WebGPU).

## Related

- Examples: `../examples/`.
- GPU subdir: `./gpu/`.
