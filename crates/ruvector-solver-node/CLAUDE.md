# ruvector-solver-node

Node.js NAPI-RS bindings for the RuVector sublinear-time solver. Provides high-performance sparse linear-system solving, PageRank computation, and complexity estimation. All heavy compute runs on worker threads via `tokio::task::spawn_blocking` so the Node event loop is never blocked.

Published as `@ruvector/solver` (see `package.json`).

## Layout

- `Cargo.toml` — `cdylib` only. Path dep on `ruvector-solver` (no default features). `napi`, `napi-derive`, `tokio`, `serde`, `serde_json`. Lints relaxed (research tier).
- `build.rs` — invokes `napi-build`.
- `package.json` — npm metadata with the cross-platform target triples (`x86_64-unknown-linux-gnu/musl`, `aarch64-*`, `aarch64-apple-darwin`, `x86_64-pc-windows-msvc`).
- `src/lib.rs` — only source file.

## Public API (`#[napi]`)

- `SolveConfig { values, col_indices, row_ptrs, rows, cols, rhs, tolerance?, max_iterations?, algorithm? }` — CSR sparse matrix + RHS. Algorithm: `"neumann" | "jacobi" | "gauss-seidel" | "conjugate-gradient"` (default `"jacobi"`).
- `SolveResult { solution, iterations, residual, converged, algorithm, time_us }`.
- Additional NAPI exports for PageRank and complexity estimation.

## Related

- `crates/ruvector-solver` — Rust solver library (`ruvector_solver::types::Algorithm`).
- `crates/ruvector-graph-node`, `crates/ruvector-cluster` — sibling Node bindings.
- npm package: `@ruvector/solver`.
