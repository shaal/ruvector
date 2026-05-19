# ruvllm/src/lora

MicroLoRA fine-tuning pipeline for real-time per-request adaptation.
Ultra-light: <1MB per adapter, designed for hot-swap during inference.

## Files
- `mod.rs` - public API + quick-start docs.
- `micro_lora.rs` - `MicroLoRA`, `MicroLoraConfig` (rank, alpha, target
  modules); the runtime delta computation.
- `adapter.rs` - generic `Adapter` interface backing both MicroLoRA and
  full LoRA.
- `training.rs` - LoRA training-loop primitives.
- `adapters/` - adapter merging + trainer subcomponents (see
  `adapters/CLAUDE.md`).

## Key types
- `MicroLoRA`, `MicroLoraConfig`, `TargetModule` (QProj, KProj, ...),
  `AdaptFeedback`.
