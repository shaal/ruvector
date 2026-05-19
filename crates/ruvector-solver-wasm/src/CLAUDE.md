# ruvector-solver-wasm/src

WASM glue for the sublinear-time solver.

## Files
- `lib.rs` - `init`, `version`, `JsSolver`. `JsSolver::solve(...)` accepts
  CSR arrays directly from JS typed arrays and dispatches into the
  algorithm router (Neumann series / forward-push / CG). Returns
  JSON-serializable result via `serde`.
- `utils.rs` - `set_panic_hook`, `console_log`, and
  `csr_from_js_arrays(...)` (zero-copy CSR construction from JS).
