# ruvector-postgres/src/sparse

Sparse vector support — efficient storage and search of high-dimensional sparse embeddings (BM25, SPLADE, learned sparse).

## Files

- `mod.rs` — Re-exports `SparseVec`, `sparse_cosine`, `sparse_dot`, `sparse_euclidean`.
- `types.rs` — `SparseVec` (COO format storage).
- `distance.rs` — Sparse-sparse distance kernels.
- `operators.rs` — pgrx SQL operators.
- `tests.rs` — Inline unit tests.

## Pointers

- See `../../docs/guides/SPARSE_QUICKSTART.md`, `../../docs/guides/SPARSE_VECTORS.md`, `../../SPARSE_DELIVERY.md`.
