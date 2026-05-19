# ruvector-diskann/src

Flat-file source layout.

## Files

- `lib.rs` — module declarations + top-level re-exports.
- `error.rs` — `DiskAnnError` + Result alias.
- `distance.rs` — distance kernels (with `simsimd` fast path under `simd` feature).
- `graph.rs` — Vamana graph: greedy search, α-robust pruning, mmap layout.
- `index.rs` — `DiskAnnIndex` + `DiskAnnConfig` (the public entry point).
- `pq.rs` — `ProductQuantizer` for compressed candidate filtering.
