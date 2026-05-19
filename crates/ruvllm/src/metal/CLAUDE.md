# ruvllm/src/metal

Metal GPU acceleration for Apple Silicon M4 Pro. Provides Flash Attention
(tiled, O(N) memory), GEMM via `simdgroup_matrix`, RMSNorm/LayerNorm with
warp-level reductions, and RoPE. Optimized for 16KB threadgroup memory and
FP16 throughput.

## Files
- `mod.rs` - public API + M4 Pro optimization notes.
- `context.rs` - Metal device/queue context.
- `buffers.rs` - buffer allocation, lifecycle, and reuse.
- `pipelines.rs` - Metal pipeline state objects (loaded from compiled
  shaders).
- `operations.rs` - dispatch wrappers for the public ops.
- `shaders/` - `.metal` shader source; see `shaders/CLAUDE.md`.
