# ruvector-coherence

Coherence-measurement proxies for comparing attention mechanisms — metrics, comparison, quality guardrails, and batched evaluation. Plus an opt-in spectral-coherence scorer for graph-index health (behind `spectral` feature).

## Layout

- `Cargo.toml` — minimal deps (`serde`, `serde_json`). Feature `spectral` enables `spectral.rs`.
- `src/lib.rs` — module declarations + re-exports of all major public types.
- `src/metrics.rs` — `contradiction_rate`, `entailment_consistency`, `delta_behavior`, `DeltaMetric`.
- `src/comparison.rs` — `compare_attention_masks`, `edge_flip_count`, `jaccard_similarity`, `ComparisonResult`.
- `src/quality.rs` — `cosine_similarity`, `l2_distance`, `quality_check`, `QualityResult`.
- `src/batch.rs` — `evaluate_batch`, `BatchResult` for batched mechanism comparison.
- `src/spectral.rs` (feature `spectral`) — `SpectralCoherenceScore`, `SpectralTracker`, `HnswHealthMonitor`, `HealthAlert`, `CsrMatrixView`, plus estimators: `estimate_fiedler`, `estimate_spectral_gap`, `estimate_largest_eigenvalue`, `estimate_effective_resistance_sampled`, `compute_degree_regularity`.

## Tests

- `tests/spectral_bench.rs` — spectral path bench-style integration test.

## Related crates

- `crates/ruvector-attention` — measured side; comparison + spectral helpers feed back into attention selection.
- `crates/prime-radiant/src/coherence` — uses similar concepts at engine scale.
