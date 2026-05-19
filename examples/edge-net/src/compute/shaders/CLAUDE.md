# edge-net/src/compute/shaders

GPU shader sources embedded into the WASM via `include_str!`.

## Important files
- `matmul.wgsl` — WebGPU matmul.
- `matmul.frag` — WebGL2 matmul fragment shader.
- `attention.wgsl` — WebGPU attention kernel.
- `lora.wgsl` — WebGPU LoRA kernel.

## Related
- Consumers: `../webgpu.rs`, `../webgl_compute.rs`.
