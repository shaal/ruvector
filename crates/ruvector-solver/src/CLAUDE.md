# ruvector-solver/src

Sublinear sparse-linear-solver implementation.

## Top-level

- `lib.rs` — crate doc + module declarations.
- `types.rs` — `CsrMatrix<T>` (COO-build + CSR storage), `ComputeBudget`.
- `traits.rs` — `SolverEngine` trait (uniform `solve(&matrix, &rhs)` interface).
- `error.rs` — `SolverError`.
- `router.rs` — selects the best algorithm given problem shape + budget.

## Algorithm implementations (feature-gated)

- `neumann.rs` — Neumann series (default).
- `forward_push.rs` — forward push (default).
- `backward_push.rs` — backward push.
- `random_walk.rs` — hybrid random walk.
- `cg.rs` — conjugate gradient (default).
- `bmssp.rs` — BMSSP.
- `true_solver.rs` — reference exact solver.

## Infrastructure

- `budget.rs` — compute-budget enforcement (iter caps, timeouts).
- `arena.rs` — bump arena.
- `simd.rs` — SIMD inner loops.
- `events.rs` — solver event emission for observability.
- `audit.rs` — audit-trail support.
- `validation.rs` — post-solve validation (residual checks).
