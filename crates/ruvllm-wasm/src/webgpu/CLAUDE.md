# ruvllm-wasm/src/webgpu

WebGPU acceleration path (feature `webgpu`). Compiles shaders + manages buffers + dispatches compute pipelines.

## Files

- `mod.rs` — module entry; gated on `webgpu` feature.
- `compute.rs` — compute pipeline + dispatch.
- `buffers.rs` — `GpuBuffer` allocation, upload/download.
- `shaders.rs` — shader-module compilation (loads WGSL from `shaders/`).

## Subdirectories

- `shaders/` — WGSL compute shaders.
