# ruvllm/src/intelligence

Trait-based extension point for external systems to feed quality signals
into RuvLLM's learning loops (SONA, embedding classifier, model-router
calibration). External providers register with `IntelligenceLoader`, which
fans signals out to the consumers.

## Files
- `mod.rs` - the entire module: `IntelligenceLoader` plus the
  `IntelligenceProvider` trait and ingest plumbing for downstream learners.
