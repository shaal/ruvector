# ruvllm/tests

Standalone integration tests for the LLM runtime (separate cargo targets,
public API only). Internal mod-level tests live in `../src/tests/`.

## Files
- `acceptance_gates.rs` - acceptance criteria for releases.
- `adapter_integration.rs` - LoRA adapter integration.
- `ane_integration.rs`, `ane_test_utils.rs` - Apple Neural Engine path.
- `autodetect_integration.rs` - hardware autodetect.
- `backend_integration.rs` - backend dispatcher.
- `cross_platform.rs`, `cross_platform_v21.rs` - cross-platform parity.
- `e2e_integration.rs`, `e2e_integration_test.rs` - end-to-end inference.
- `gguf_integration.rs`, `gguf_loader_test.rs` - GGUF loader.
- `hadamard_tests.rs` - Hadamard / QuIP correctness.
- `kernel_integration.rs` - kernel dispatch.
- `lora_integration.rs` - LoRA hot-swap.
- `mistral_backend_test.rs` - mistral-rs backend integration.
- `model_arch_integration.rs` - model architecture tests.
- `moe_integration.rs` - MoE routing.
- `pi_quant_tests.rs` - Pi-quantization correctness.
- `real_model_test.rs` - tests against real downloaded models.
- `ruvltra_e2e.rs`, `ruvltra_tests.rs` - RuvLTRA-specific tests.
- `serving_integration.rs` - continuous-batching serving engine.
- `simd_equivalence_tests.rs` - scalar vs SIMD equivalence.
- `sona_integration.rs` - SONA learning loop integration.
- `speculative_integration.rs` - speculative decoding.
- `ste_tests.rs` - straight-through estimator (QAT).
- `fixtures/` - test fixtures (see `fixtures/CLAUDE.md`).
