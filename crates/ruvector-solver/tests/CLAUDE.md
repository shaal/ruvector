# ruvector-solver/tests

Integration tests (`cargo test --test <name>`).

## Files

- `helpers.rs` — shared test helpers (compiled into each test target via `mod helpers;`).
- `test_csr_matrix.rs` — `CsrMatrix` construction and access.
- `test_neumann.rs` — Neumann-series solver correctness.
- `test_push.rs` — forward / backward push correctness.
- `test_cg.rs` — conjugate-gradient correctness.
- `test_router.rs` — algorithm router selection logic.
- `test_validation.rs` — post-solve validation (residual norms, divergence detection).
