# ruvllm

LLM serving runtime with Ruvector integration. Edge-focused inference across
heterogeneous hardware. Uses Ruvector as a unified memory layer with three
roles: policy memory store, session-state index, and witness-log index.
Integrates SONA learning for continuous self-improvement.

## Important files
- `Cargo.toml` - large workspace member. Default-features deps on
  `ruvector-core` (with storage/hnsw/parallel/simd) and `ruvector-sona`;
  optional `ruvector-attention`, `ruvector-graph`, `ruvector-gnn`. Other
  notable deps: `ndarray`, `dashmap`, `bincode`, `regex`, optional `tokio`,
  optional `rayon`.
- `CHANGELOG.md` - version history.
- `.reasoning_bank_patterns` - persisted ReasoningBank pattern store
  (binary; do not edit by hand).
- `src/lib.rs` - top-level docs, lint allows, public re-exports.

## Top-level src files
- `error.rs`, `types.rs` - errors and shared types.
- `kv_cache.rs` - `TwoTierKvCache` (FP16 tail + quantized store).
- `paged_attention.rs` - `PagedAttention` (memory-efficient attention with
  page tables).
- `adapter_manager.rs` - `AdapterManager` (LoRA hot-swap).
- `policy_store.rs` - Ruvector-backed policy store.
- `session.rs`, `session_index.rs` - session lifecycle + multi-turn index.
- `witness_log.rs` - audit log with HNSW-indexed semantic search.
- `autodetect.rs`, `capabilities.rs`, `tokenizer.rs`, `memory_pool.rs`,
  `speculative.rs`, `ruvector_integration.rs` - support.

## Source modules (each has its own CLAUDE.md)
- `backends/` - Candle, mistral-rs, CoreML; Gemma2, Mistral, Phi3 wrappers.
- `bitnet/` - BitNet b1.58 ternary quantization.
- `claude_flow/` - Claude Flow integration: agent routing, model routing,
  HNSW router, hooks, pretrain pipeline, Claude API streaming.
- `context/` - intelligent context manager: agentic / episodic / semantic /
  working memory.
- `evaluation/` - three-layer eval harness (correctness, diff quality,
  economics).
- `gguf/` - llama.cpp GGUF v3 loader (Q4_0..Q8_K, F16/F32).
- `hub/` - HuggingFace Hub upload/download with progress.
- `intelligence/` - external intelligence providers feeding SONA signals.
- `kernels/` - NEON-optimized LLM kernels for Apple Silicon.
- `lora/` - MicroLoRA fine-tuning (<1MB per adapter) + adapters/.
- `metal/` - Metal GPU acceleration (Flash Attention, GEMM, RMSNorm, RoPE)
  + `metal/shaders/` (.metal source).
- `models/` - RuvLTRA-Small (Qwen 0.5B) and RuvLTRA-Medium (Qwen2.5-3B).
- `moe/` - Mixture-of-Experts routing + cache (ADR-092).
- `optimization/` - real-time optimizer, SONA-LLM integration, metrics.
- `qat/` - Quantization-Aware Training (STE, distillation, LoRA-QAT).
- `quality/` - multi-dimensional quality scoring.
- `quantize/` - Pi-quantization, QuIP, Hadamard, security.
- `reasoning_bank/` - trajectory recording, pattern store, verdicts,
  consolidation, distillation.
- `reflection/` - self-reflection (IoE, multi-perspective, error recovery).
- `serving/` - continuous batching engine + scheduler + KV cache manager.
- `sona/` - SONA learning integration + RuvLTRA pretraining.
- `training/` - Claude dataset, GRPO, MCP tool datasets, contrastive.
- `tests/` - internal in-tree tests.

## Tests / benches / examples
- `benches/` - criterion benches for attention, RoPE, matmul, norm, MoE,
  serving, LoRA, ANE, Metal, pi-quant, turbo-quant, e2e, RuvLTRA.
- `tests/` - integration tests for adapters, GGUF, kernels, model
  architectures, LoRA, MoE, Metal, SONA, RuvLTRA e2e, etc.
- `tests/fixtures/` - `test_prompts.json`, `perplexity_baselines.json`.
- `examples/` - `benchmark_model`, `download_test_model`, `run_eval`,
  `generate_claude_dataset`, `train_contrastive`, `hub_cli`.
- `models/ruvltra_small.json` - shipped model spec.
- `docs/` - GitHub issue drafts for SOTA/mistral-rs/V2 features.

## Related
- `ruvllm-cli` - the user-facing CLI (`ruvllm` binary).
- `../sona` - SONA learning library.
- `../ruvector-core`, `../ruvector-attention`, `../ruvector-graph`,
  `../ruvector-gnn`.
