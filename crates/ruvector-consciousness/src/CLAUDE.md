# ruvector-consciousness/src

All algorithm implementations live here as top-level modules (no nested subdirs).

## Φ (integrated information)

- `phi.rs` — primary Φ entrypoint and auto-selecting algorithm dispatcher.
- `iit4.rs` — IIT 4.0 axiom-faithful implementation.
- `phi_id.rs` — Φ-ID (information-decomposition variant).
- `chebyshev_phi.rs` — Chebyshev polynomial spectral approximation O(n² log n).
- `coherence_phi.rs` — accelerated by `ruvector-coherence` (feature `coherence-accel`).
- `mincut_phi.rs` — accelerated by `ruvector-mincut` (feature `mincut-accel`).
- `witness_phi.rs` — Φ paired with cognitive-container witness (feature `witness`).

## Emergence / causal information

- `emergence.rs` — effective information / causal emergence O(n³).
- `rsvd_emergence.rs` — randomized SVD acceleration.
- `pid.rs` — Partial Information Decomposition.

## MIP / partition search

- `collapse.rs` — quantum-inspired collapse search.
- `geomip.rs` — geometric MIP search.

## Qualia / structure

- `ces.rs` — Cause-Effect Structure.

## Streaming / bounds

- `streaming.rs` — incremental / online Φ.
- `bounds.rs` — upper/lower-bound estimators.

## Perf primitives

- `simd.rs` — AVX2 KL-divergence, entropy, dense matvec.
- `sparse_accel.rs` — sparse-matrix kernels.
- `parallel.rs` — rayon-based parallel variants (feature `parallel`).
- `arena.rs` — zero-alloc bump arena for hot paths.

## Shared

- `types.rs`, `traits.rs`, `error.rs` — common types/traits/errors.
- `lib.rs` — module declarations and `#![allow(...)]` workspace lints.
