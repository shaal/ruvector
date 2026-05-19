# ruvector-cnn/docs

Design + implementation notes for the CNN crate (ADR-091 series + INT8 plumbing).

## Files

- `ADR-091-PHASE-2.1-COMPLETE.md`
- `ADR-091-PHASE-3-IMPLEMENTATION.md`
- `ADR-091-PHASE-4-IMPLEMENTATION.md`
- `ADR-091_PHASE_6_SUMMARY.md`
- `GRAPH_REWRITE_SUMMARY.md` — overview of the INT8 graph-rewrite pass (consumed by `src/quantize/graph_rewrite.rs`).
- `INT8_KERNELS_IMPLEMENTATION.md` — how the per-arch INT8 kernels are organised.
- `INT8_QUANTIZATION_DESIGN.md` — quantization scheme + calibration design.
- `QUANTIZED_LAYERS_USAGE.md` — usage notes for the quantized layer set.
