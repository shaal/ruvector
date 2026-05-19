# ruvector-sparsifier

Dynamic spectral graph sparsification: maintains a small weighted shadow graph **H** that preserves the Laplacian energy of a full
graph **G** within `(1 +/- epsilon)`. Implements the ADKKP16 approach adapted for real-time use: spanning-forest backbone +
random-walk effective-resistance importance + spectral sampling + periodic quadratic-form audits.

## Files

- `Cargo.toml` — depends on serde, thiserror, tracing, rand, parking_lot, dashmap, rayon, ordered-float. Dev: criterion,
  proptest, approx. Single bench `sparsifier_bench`. `README.md` referenced.
- `src/` — implementation (see `src/CLAUDE.md`).
- `tests/integration_tests.rs` — end-to-end sparsifier correctness/quality tests.
- `benches/sparsifier_bench.rs` — criterion microbench.

## Features

- `default = ["static-sparsify", "dynamic"]`.
- `static-sparsify` — build-once sparsification.
- `dynamic` — incremental insert/delete with audit drift detection.
- `simd` — SIMD distance/laplacian kernels.
- `wasm` — wasm-compatible build.
- `audit` — explicit audit hooks.
- `full = ["static-sparsify", "dynamic", "simd", "audit"]`.

## Public API surface

`AdaptiveGeoSpar`, `SparseGraph`, `SparsifierConfig`, `Sparsifier` (trait), plus types for audit reports and importance scores.

## Related

- `../ruvector-graph`, `../ruvector-gnn` (likely) — consumers operating on the compressed graph H.
- `../ruvector-core` — vector primitives if integrated.
