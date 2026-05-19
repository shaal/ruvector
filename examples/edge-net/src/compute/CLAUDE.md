# edge-net/src/compute

Compute backends for executing tasks in-browser: SIMD, Web Workers, WebGL, WebGPU. Provides a `Tensor` abstraction and a backend trait so the scheduler can pick the fastest available path.

## Important files
- `mod.rs` — module entry.
- `backend.rs` / `backends.rs` — backend trait + selection.
- `tensor.rs` / `types.rs` — tensor types.
- `simd.rs` — WASM SIMD path.
- `workers.rs` — Web Workers backend.
- `webgl_compute.rs` — WebGL2 backend.
- `webgpu.rs` — WebGPU backend.
- `shaders/` — GLSL/WGSL shader sources used by the GL/GPU backends.
