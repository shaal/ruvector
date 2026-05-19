# ruvector-postgres/src/solver

Solver integration module — exposes `ruvector-solver` as SQL functions.

## Files

- `mod.rs` — Helper `edges_json_to_csr(json)` converts a JSON edge-list (`[[src, dst]]` or `[[src, dst, weight]]`) to a `CsrMatrix<f64>`. Declares `operators` submodule.
- `operators.rs` — pgrx SQL functions for graph linear algebra (PageRank, Laplacians, etc.) via the solver crate.
