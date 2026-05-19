# ruvllm/src/metal/shaders

Metal shader sources (`.metal`) compiled and loaded by `../pipelines.rs`.
Optimized for Apple Silicon M4 Pro (simdgroup matrices, FP16, 16KB
threadgroup memory).

## Files
- `attention.metal` - basic attention kernel.
- `attention_fused.metal` - fused softmax+matmul attention.
- `rope_attention.metal` - fused RoPE + attention.
- `rope.rs` - standalone RoPE shader.
- `norm.metal` - RMSNorm / LayerNorm kernels.
- `gemm.metal` - matrix-matrix multiply using `simdgroup_matrix`.
- `gemv.metal` - matrix-vector multiply.
- `quantized.metal` - quantized matmul / dequant.
- `fused_ops.metal` - misc fused ops (e.g. activation+norm).
