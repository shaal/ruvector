# ruvllm/src/bitnet

Microsoft Research BitNet b1.58 ternary weight quantization for the
Craftsman Ultra 30b 1bit model. Post-training quantization (PTQ) of FP16
weights to {-1, 0, +1} using absmean quantization. Enables
multiplication-free inference at ~2 bits/weight.

## Files
- `mod.rs` - public API + crate-style docs.
- `quantizer.rs` - `quantize_tensor` (absmean PTQ); `PtBitnetConfig`.
- `dequantize.rs` - `dequantize_bitnet_t158` (pack -> FP32 for validation).
- `backend.rs` - inference backend integration for ternary weights.
- `expert_cache.rs` - cache for ternary MoE experts.
- `gguf_export.rs` - export ternary models in GGUF.
- `eval.rs` - evaluation/perplexity helpers for ternary checkpoints.
- `TEST_COVERAGE.md` - test-coverage notes for the module.

## Key types
- `TernaryTensor` - 2-bit packed ternary weights.
- `PtBitnetConfig` - PTQ configuration.
