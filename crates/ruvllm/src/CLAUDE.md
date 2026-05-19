# ruvllm/src

Source tree for the LLM serving runtime. `lib.rs` re-exports the public API;
top-level files own cross-cutting infra; subdirectories own bounded
domains.

## Top-level files
- `lib.rs` - crate docs + lint overrides + public re-exports.
- `error.rs`, `types.rs` - error enum and shared value types.
- `kv_cache.rs` - `TwoTierKvCache` (FP16 tail + quantized store).
- `paged_attention.rs` - `PagedAttention` engine with page tables.
- `adapter_manager.rs` - LoRA hot-swap orchestration.
- `policy_store.rs` - Ruvector-backed policy + threshold store with
  semantic search.
- `session.rs`, `session_index.rs` - session lifecycle and multi-turn
  HNSW state index.
- `witness_log.rs` - audit log with HNSW-indexed semantic search.
- `autodetect.rs` - hardware autodetection.
- `capabilities.rs` - feature/capability flags.
- `tokenizer.rs` - tokenization wrappers.
- `memory_pool.rs` - shared memory pooling.
- `speculative.rs` - speculative decoding harness.
- `ruvector_integration.rs` - bridge utilities into the Ruvector stack.

## Submodules
- `backends/` - Candle / mistral-rs / CoreML; model wrappers (Mistral,
  Gemma2, Phi3); hybrid pipeline.
- `bitnet/` - BitNet b1.58 ternary quantization.
- `claude_flow/` - Claude Flow integration (agent + model routing, hooks,
  pretrain).
- `context/` - context manager (agentic / episodic / semantic / working).
- `evaluation/` - three-layer evaluation harness.
- `gguf/` - GGUF v3 loader + quantization decoders.
- `hub/` - HuggingFace Hub upload / download.
- `intelligence/` - external intelligence-provider trait.
- `kernels/` - NEON-optimized kernels for Apple Silicon.
- `lora/` - MicroLoRA + adapters/.
- `metal/` - Metal GPU acceleration + `shaders/` (.metal source).
- `models/` - RuvLTRA model architectures.
- `moe/` - Mixture-of-Experts routing/caching.
- `optimization/` - real-time optimizer + SONA-LLM glue.
- `qat/` - Quantization-Aware Training.
- `quality/` - quality scoring engine.
- `quantize/` - Pi/QuIP/Hadamard quantization.
- `reasoning_bank/` - ReasoningBank (trajectories, verdicts, patterns).
- `reflection/` - self-reflection (IoE, multi-perspective).
- `serving/` - continuous batching engine + scheduler.
- `sona/` - SONA learning integration.
- `tests/` - in-tree unit/integration tests.
- `training/` - dataset generation + GRPO training.
