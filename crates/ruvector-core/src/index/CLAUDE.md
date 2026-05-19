# ruvector-core/src/index

Index implementations behind the `Index` trait (see sibling `index.rs`).

## Files

- `flat.rs` — exact brute-force flat index (baseline / small datasets / ground truth).
- `hnsw.rs` — HNSW approximate index built on `hnsw_rs`; primary production index.
