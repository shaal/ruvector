# prime-radiant/src/gpu

GPU acceleration via `wgpu` for coherence computations: parallel residuals, restriction-map matrix ops, atomic energy aggregation, and power-iteration spectral analysis.

## Files

- `mod.rs` — module entry with the GPU pipeline diagram.
- `device.rs` — `GpuDevice`: instance/adapter/device/queue init.
- `buffer.rs` — `GpuBuffer` + buffer pool for upload/download.
- `pipeline.rs` — pipeline cache + bind-group layout.
- `dispatch.rs` — `GpuDispatcher`: kernel dispatch & synchronization.
- `kernels.rs` — high-level kernel entry points.
- `engine.rs` — GPU-backed coherence engine variant.
- `error.rs` — GPU-specific errors.
- `shaders/` — WGSL compute shaders.

## Feature flag

Behind the GPU feature; see `Cargo.toml` for the wgpu version pin.
