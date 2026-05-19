# ruvllm/src/qat

Quantization-Aware Training (ADR-090 Phase 2). Training with quantization in
the loop, preserving ~90% of reasoning at 2-3 bit precision vs ~40% for PTQ.

## Files
- `mod.rs` - public API + module map.
- `config.rs` - `QatConfig`, `SteVariant`, `QuantGranularity`.
- `ste.rs` - Straight-Through Estimator variants.
- `differentiable_quant.rs` - `DifferentiableQuantizer` trait + impls.
- `calibration.rs` - `CalibrationEngine` (scale init from calibration data).
- `distillation.rs` - knowledge-distillation loss (L_task + L_KD).
- `reasoning_loss.rs` - chain-of-thought fidelity loss.
- `lora_qat.rs` - LoRA + QAT integration.
- (training_loop is referenced in module docs; orchestration sits inline
  alongside the above.)
