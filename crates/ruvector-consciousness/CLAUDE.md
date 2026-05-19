# ruvector-consciousness

SOTA consciousness metrics: IIT Φ (Integrated Information) computation, causal emergence / effective information, and quantum-inspired MIP search. Provides exact, spectral, and stochastic Φ algorithms with SIMD acceleration, zero-alloc arena hot paths, and auto-selection by system size.

## Important files

- `Cargo.toml` — broad feature surface (`phi`, `emergence`, `collapse`, `simd`, `wasm`, `parallel`, plus accel features pulling in `ruvector-solver`, `ruvector-sparsifier`, `ruvector-mincut`, `ruvector-math`, `ruvector-coherence`, `ruvector-cognitive-container`).
- `src/lib.rs` — crate doc with algorithm complexity table; module declarations.
- `benches/phi_benchmark.rs` — benchmark for Φ algorithms.
- `tests/integration.rs` — cross-module integration tests.

## Module map (src/)

- `phi.rs`, `iit4.rs`, `phi_id.rs` — IIT Φ implementations (exact + IIT 4.0 axioms + Partial Information Decomposition).
- `chebyshev_phi.rs` — Chebyshev-polynomial spectral Φ approximation.
- `coherence_phi.rs`, `witness_phi.rs`, `mincut_phi.rs` — Φ variants accelerated by coherence/witness/mincut crates.
- `emergence.rs`, `rsvd_emergence.rs` — causal emergence / effective information (incl. randomized SVD variant).
- `ces.rs` — Cause-Effect Structure (qualia geometry).
- `collapse.rs` — quantum-inspired MIP search O(√N · n²).
- `pid.rs` — Partial Information Decomposition.
- `geomip.rs` — geometric MIP partition search.
- `streaming.rs` — incremental / online Φ estimation.
- `bounds.rs` — upper/lower bounds on Φ.
- `simd.rs`, `sparse_accel.rs`, `parallel.rs`, `arena.rs` — perf primitives (AVX2 KL/entropy, sparse matmul, rayon parallel, bump arena).
- `types.rs`, `traits.rs`, `error.rs` — shared types / traits / error enum.

## Features

`default = ["phi", "emergence", "collapse"]`; `full` enables everything (parallel + all accel + witness). Accel features only compile their respective adapters when enabled.

## Public API

`Phi`, `EmergenceMetric`, `CollapseSearch`, `StreamingPhi`, plus algorithm structs under each module.

## Related

- `crates/ruvector-solver`, `crates/ruvector-sparsifier`, `crates/ruvector-mincut`, `crates/ruvector-math`, `crates/ruvector-coherence`, `crates/ruvector-cognitive-container` (optional accel deps).
