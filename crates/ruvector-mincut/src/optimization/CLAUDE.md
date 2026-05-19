# ruvector-mincut/src/optimization

Performance optimizations layered onto the core algorithms.

## Files

- `mod.rs` — module wiring.
- `cache.rs` — cut-value cache.
- `dspar.rs` — dynamic sparsifier optimization.
- `parallel.rs` — parallel update strategies (`rayon`/`crossbeam`).
- `pool.rs` — object pooling for hot-path allocations.
- `simd_distance.rs` — SIMD-accelerated distance helpers.
- `wasm_batch.rs` — wasm-friendly batched-update path.
- `benchmark.rs` — internal benchmark helpers shared by `benches/*`.
