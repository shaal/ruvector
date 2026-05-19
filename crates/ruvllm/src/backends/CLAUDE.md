# ruvllm/src/backends

Pluggable LLM inference backends. Currently: Candle (Rust HuggingFace) with
Metal accel, mistral-rs (PagedAttention + X-LoRA), and CoreML (Apple Neural
Engine). Supports Mistral (7B, Codestral), Llama (1B-70B; L2/L3), Phi
(1.5/2/3); GGUF quantization Q4_0/Q4_1/Q4_K, Q8_0/Q8_1.

## Files
- `mod.rs` - backend trait + dispatcher; selects backend by capability /
  config.
- `candle_backend.rs` - Candle (Rust HuggingFace) backend with Metal.
- `coreml_backend.rs` - CoreML / Apple Neural Engine backend.
- `mistral_backend.rs` - mistral-rs backend with PagedAttention + X-LoRA.
- `gemma2.rs` - Gemma2 architecture wrapper.
- `phi3.rs` - Phi3 architecture wrapper.
- `mistral.rs` (referenced via `mistral_backend`) and adjacent files share
  config patterns.
- `hybrid_pipeline.rs` - composed pipeline that can mix backends per stage.
