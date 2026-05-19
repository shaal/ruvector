# ruvector-rairs/src

Flat-file source layout. `#![forbid(unsafe_code)]`, `#![warn(missing_docs)]`.

## Files

- `lib.rs` — crate doc + module declarations + re-exports.
- `main.rs` — `rairs-demo` binary (the de-facto benchmark per the provenance note).
- `error.rs` — `RairsError`.
- `index.rs` — `AnnIndex` trait + `SearchResult`.
- `ivf.rs` — `IvfFlat` (baseline IVF, single assignment).
- `kmeans.rs` — k-means centroid clustering.
- `rairs.rs` — `RairsStrict` (dual RAIR, flat layout).
- `seil.rs` — `RairsSeil` (dual RAIR with SEIL 32-vector dedup blocks).
