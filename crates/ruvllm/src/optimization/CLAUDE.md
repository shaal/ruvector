# ruvllm/src/optimization

Real-time optimization infrastructure for LLM inference. Integrates SONA
learning with MicroLoRA and custom kernels.

## Files
- `mod.rs` - public API + quick-start showing `SonaLlm`, `RealtimeOptimizer`,
  `MetricsCollector`, `ConsolidationStrategy`.
- `sona_llm.rs` - `SonaLlm` / `SonaLlmConfig` integration (instant /
  background / consolidation tiers).
- `realtime.rs` - `RealtimeOptimizer` driving online updates.
- `metrics.rs` - `MetricsCollector` for instant feedback.
