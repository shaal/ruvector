# onnx-embeddings/src/gpu

GPU backend for ONNX embedding generation, built on `wgpu` (WebGPU/Vulkan/Metal/DX12). Optional via the `gpu` feature.

## Files

- `mod.rs` - Module surface and backend dispatch.
- `backend.rs` (~46KB) - Core GPU backend: device init, buffer mgmt, pipeline cache.
- `config.rs` - GPU config types (adapter selection, workgroup sizes).
- `operations.rs` (~35KB) - High-level ops (matmul, layer norm, pooling) composed from shaders.
- `shaders.rs` - WGSL shader sources.
- `tests.rs` - Inline GPU tests.

## How to build

```bash
cargo build --features gpu
cargo bench --features gpu --bench gpu_benchmark
```

## Related

- Docs: `../../docs/GPU_ACCELERATION.md`.
- Embedder: `../embedder.rs`.
