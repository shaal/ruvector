# ruvector-nervous-system/src/plasticity

Learning / plasticity rules. See `docs/EWC_IMPLEMENTATION.md`.

## Files

- `mod.rs` — façade.
- `btsp.rs` — Behavioral Time-Series Pattern plasticity (benchmarked in `benches/btsp_bench.rs`).
- `eprop.rs` — e-prop online recurrent learning (benchmarked in `benches/eprop_bench.rs`).
- `consolidate.rs` — Elastic Weight Consolidation (EWC); parallelised under `parallel` feature.
