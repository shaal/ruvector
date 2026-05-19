# 04-sparse-persistent-homology/src

## Files

- `lib.rs` — re-exports.
- `apparent_pairs.rs` — apparent-pairs short-circuit to skip most
  reduction work.
- `sparse_boundary.rs` — sparse boundary matrix data structures.
- `simd_filtration.rs` — SIMD-accelerated filtration sorting.
- `simd_matrix_ops.rs` — SIMD column / reduction kernels.
- `streaming_homology.rs` — incremental / streaming filtration so the
  diagram updates as new simplices arrive.

## Related

- `../benches/sparse_homology_bench.rs`
- `../complexity_analysis.md` — sub-cubic argument
