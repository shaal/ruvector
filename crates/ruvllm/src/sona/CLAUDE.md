# ruvllm/src/sona

SONA (Self-Optimizing Neural Architecture) learning integration for the
RuvLLM runtime. Provides the three-tier learning loop (Instant /
Background / Deep) and RuvLTRA-specific pretraining presets.

## Files
- `mod.rs` - public API + learning-loop architecture.
- `integration.rs` - `SonaIntegration` (runtime learning during inference).
- `ruvltra_pretrain.rs` - `RuvLtraPretrainer` (pretraining configurations
  for RuvLTRA-Small/Medium).
