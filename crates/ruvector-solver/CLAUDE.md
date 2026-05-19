# ruvector-solver

Sublinear-time iterative sparse linear solvers for the RuVector ecosystem. Solves `Ax = b` for sparse CSR matrices using Neumann series, forward/backward push, hybrid random walk, conjugate gradient, BMSSP, and a "true solver" variant. Targets O(log n) to O(√n) algorithms for PageRank and spectral methods.

## Important files

- `Cargo.toml` — feature-gated solver selection: `neumann` / `forward-push` / `backward-push` / `hybrid-random-walk` / `true-solver` / `cg` / `bmssp` / `all-algorithms`; plus `parallel`, `simd`, `wasm`, `nalgebra-backend`. Default: `neumann`, `cg`, `forward-push`.
- `build.rs` — build script (probably feature/cfg detection).
- `src/lib.rs` — doc + worked example with `NeumannSolver` on a CSR matrix.

## Module map (src/)

- `lib.rs` — module decls.
- `types.rs` — `CsrMatrix<T>`, `ComputeBudget`, shared types.
- `traits.rs` — `SolverEngine` trait.
- `error.rs` — `SolverError`.
- `router.rs` — automatic algorithm selection / routing.
- `neumann.rs` — Neumann-series solver (`neumann`).
- `forward_push.rs`, `backward_push.rs` — push-based local solvers (`forward-push`, `backward-push`).
- `random_walk.rs` — hybrid random walk (`hybrid-random-walk`).
- `cg.rs` — conjugate gradient (`cg`).
- `bmssp.rs` — BMSSP solver (`bmssp`, security-reviewed in `crates/ruvector-mincut/docs/security/`).
- `true_solver.rs` — exact "true solver" reference (`true-solver`).
- `budget.rs` — `ComputeBudget` enforcement.
- `arena.rs` — bump arena for hot path.
- `simd.rs` — SIMD inner loops (`simd`).
- `events.rs`, `audit.rs`, `validation.rs` — observability and result validation.

## Tests & benches

- `tests/` — `helpers.rs`, `test_cg.rs`, `test_csr_matrix.rs`, `test_neumann.rs`, `test_push.rs`, `test_router.rs`, `test_validation.rs`.
- `benches/` — `solver_baseline.rs`, `solver_cg.rs`, `solver_e2e.rs`, `solver_neumann.rs`, `solver_push.rs`.

## Public API

`SolverEngine`, `NeumannSolver`, `CgSolver`, `ForwardPushSolver` etc., `CsrMatrix`, `ComputeBudget`, `SolverError`.

## Related

- `crates/ruvector-consciousness` consumes this via `solver-accel` feature.
- `crates/mcp-brain-server` enables the `forward-push` feature.
