# ruvllm/src/kernels

NEON-optimized LLM kernels for Apple Silicon (M1/M2/M3/M4). Public free
functions chosen by autodetect (`is_neon_available`); the Metal GPU and ANE
paths live in `../metal/` and `../backends/coreml_backend.rs` respectively.

## Files
- `mod.rs` - public API + quick-start docs (`is_neon_available`,
  `AttentionConfig`).
- `attention.rs` - `flash_attention_neon` plus attention helpers.
- `activations.rs` - SiLU/GeLU/SwiGLU/etc.
- `matmul.rs` - NEON matmul / GEMV.
- `norm.rs` - `rms_norm_neon`, layer norm.
- `quantized.rs` - quantized matmul/dequant paths.
- `accelerate.rs` - Apple Accelerate framework bridging.
- `ane_ops.rs` - Apple Neural Engine op wrappers used during dispatch.
