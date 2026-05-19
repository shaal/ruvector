# ruvector-attention/src/attention

Core attention kernels.

## Files

- `mod.rs` — module entry + re-exports.
- `scaled_dot_product.rs` — `ScaledDotProductAttention`: classic Q*K^T / sqrt(d) softmax.
- `multi_head.rs` — multi-head wrapper (parallel heads via rayon).
- `flash.rs` — FlashAttention-3 IO-aware tiled kernel.
- `mla.rs` — Multi-Head Latent Attention with KV-cache compression.
- `kv_cache.rs` — KV cache abstraction used by MLA + speculative decoding.
- `ssm.rs` — selective state-space (Mamba) attention.
- `speculative.rs` — draft/verify speculative decoding.
