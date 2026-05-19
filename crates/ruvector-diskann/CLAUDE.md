# ruvector-diskann

DiskANN / Vamana implementation for billion-scale approximate nearest neighbor search on a single node, optimised for SSD-friendly memory-mapped access.

Algorithm: Vamana graph (greedy search + α-robust pruning, bounded out-degree) + Product Quantization (PQ) for compressed candidate filtering, with memory-mapped graph data so only neighbors are loaded on demand.

Reference: Subramanya et al., "DiskANN" (NeurIPS 2019).

## Layout

- `Cargo.toml` — deps: `memmap2`, `rayon`, `serde`, `bincode`, `thiserror`, `rand`, `parking_lot`, `bytemuck`, optional `simsimd`. Features: `simd` (enables `simsimd`), `gpu` (stub for CUDA/Metal).
- `src/lib.rs` — re-exports `{DiskAnnError, Result}`, `{DiskAnnConfig, DiskAnnIndex}`, `{ProductQuantizer}`.
- `src/error.rs` — `DiskAnnError`.
- `src/distance.rs` — distance kernels (uses `simsimd` under `simd`).
- `src/graph.rs` — Vamana graph + α-robust pruning logic, mmap layout.
- `src/index.rs` — top-level `DiskAnnIndex` + `DiskAnnConfig`.
- `src/pq.rs` — `ProductQuantizer` for candidate compression.

## Tests / benches

- Dev-deps: `tempfile`. No `tests/` or `benches/` folder.

## Related crates

- Often consumed alongside `ruvector-core` and other ANN index crates (`ruvector-rabitq`, `ruvector-rairs`).
