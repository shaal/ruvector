# ruvector-coherence/src

Flat-file crate; all modules sit directly under `src/`.

## Files

- `lib.rs` — module declarations + re-exports.
- `metrics.rs` — coherence metrics (contradiction, entailment, delta).
- `comparison.rs` — attention-mask comparison (Jaccard, edge-flip).
- `quality.rs` — quality guardrails (cosine, L2, `quality_check`).
- `batch.rs` — `evaluate_batch` driver across mechanism pairs.
- `spectral.rs` (feature `spectral`) — spectral coherence scoring + HNSW health monitor.
