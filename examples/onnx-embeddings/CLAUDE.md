# onnx-embeddings

`ruvector-onnx-embeddings` (standalone crate, own workspace): pure-Rust ONNX embedding pipeline using the `ort` crate, HuggingFace `tokenizers`, with optional GPU acceleration via `wgpu`. Provides a CLI, library, RuVector integration, and a GPU backend.

## Files

- `Cargo.toml` - Standalone package; features `download-models` (default), `cuda`, `tensorrt`, `coreml`, `gpu`, `cuda-wasm`, `webgpu`.
- `Cargo.lock` - Pinned lockfile.
- `src/` - Library + CLI + GPU backend.
- `examples/` - `basic`, `batch`, `semantic_search`.
- `benches/` - `embedding_benchmark`, `gpu_benchmark` (requires `gpu`).
- `docs/GPU_ACCELERATION.md` - GPU backend design.

## How to build/run

```bash
cd /home/user/ruvector/examples/onnx-embeddings
cargo build --release
cargo run --release --example basic_embedding
cargo run --release --example semantic_search
cargo bench --bench embedding_benchmark
cargo bench --features gpu --bench gpu_benchmark
```

## Tech stack

- Rust 2021. `ort` 2.0 (ONNX Runtime), `tokenizers`, `ndarray`, `tokio`, `wgpu`+`bytemuck` (gpu).
- Release profile uses `opt-level=3`, thin LTO, single codegen unit.

## Related

- WASM sibling: `examples/onnx-embeddings-wasm` (tract-based).
- Vector store integration: `crates/ruvector-core`, `crates/ruvector-vector`.
- Embedding cache demo: `examples/rvf/examples/embedding_cache.rs`.
