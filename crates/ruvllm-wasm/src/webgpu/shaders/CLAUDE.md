# ruvllm-wasm/src/webgpu/shaders

WGSL compute shaders dispatched by `webgpu/compute.rs`.

## Files

- `attention.wgsl` — attention compute kernel.
- `matmul.wgsl` — matrix multiplication kernel (the inference workhorse).
- `softmax.wgsl` — softmax kernel.
- `norm.wgsl` — RMS / LayerNorm kernel.

Keep struct + binding layouts in sync with `webgpu/buffers.rs`.
