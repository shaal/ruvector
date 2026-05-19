# ruvector-cnn/tests

Integration / quality / kernel-equivalence tests.

## Files

- `acceptance_gates.rs` — top-level acceptance gates for the embedding pipeline.
- `backbone_test.rs` — MobileNet-V3 backbone forward + output shape (behind `backbone`).
- `contrastive_test.rs` — InfoNCE + triplet loss sanity.
- `graph_rewrite_integration.rs` — full FP→INT8 rewrite end-to-end.
- `integration_test.rs` — top-level integration with `CnnEmbedder`.
- `kernel_equivalence.rs` — scalar vs SIMD kernel parity (INT8 + FP).
- `layers_test.rs` — per-layer correctness (FP + quantized).
- `quality_validation.rs` — output-quality guardrails (cosine drift, etc.).
- `simd_test.rs` — generic SIMD path correctness.
