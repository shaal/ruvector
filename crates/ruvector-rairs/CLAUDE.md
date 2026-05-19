# ruvector-rairs

RAIRS IVF — Redundant Assignment with Amplified Inverse Residual. An Inverted File (IVF) index family that recovers the low-`nprobe` recall classic IVF loses near Voronoi-cell boundaries, by redundantly assigning each vector to a primary list and a residual-amplified secondary list, then storing the shared copies in deduplicating 32-vector blocks (SEIL layout) so the second assignment costs no extra memory.

> Provenance note (from `lib.rs`): the "RAIRS / SEIL" naming and the cited `arXiv:2601.07183 (SIGMOD 2026)` reference are NOT independently verified — treat the crate as an original implementation of the redundant-assignment idea (cf. spill lists / SOAR / multi-probe LSH) and judge it on benchmarks, not on the citation.

## Index family

| Variant | Assignment | Layout | Description |
|---|---|---|---|
| `IvfFlat` | single | flat | baseline — one list per vector |
| `RairsStrict` | dual RAIR | flat | secondary assignment, no dedup |
| `RairsSeil` | dual RAIR | SEIL | shared 32-vector blocks, query-time dedup |

All three satisfy the `AnnIndex` trait. `#![forbid(unsafe_code)]`.

## Layout

- `Cargo.toml` — deps: `rand`, `serde`; criterion bench `rairs_bench`. Defines a `rairs-demo` binary in `src/main.rs`.
- `src/lib.rs` — module declarations + re-exports: `RairsError`, `AnnIndex`, `SearchResult`, `IvfFlat`, `RairsStrict`, `RairsSeil`.
- `src/main.rs` — `rairs-demo` binary (the de-facto benchmark; see provenance note).
- `src/error.rs` — `RairsError`.
- `src/index.rs` — `AnnIndex` trait + `SearchResult` value object.
- `src/ivf.rs` — `IvfFlat` baseline.
- `src/kmeans.rs` — k-means clustering for IVF centroids.
- `src/rairs.rs` — `RairsStrict` (dual RAIR, no dedup).
- `src/seil.rs` — `RairsSeil` (dual RAIR + SEIL 32-block dedup layout).

## Benches

- `benches/rairs_bench.rs` — criterion bench.

## Related

- Design rationale: `docs/adr/ADR-193` (workspace root).
- Sibling ANN crates: `ruvector-diskann`, `ruvector-rabitq`.
