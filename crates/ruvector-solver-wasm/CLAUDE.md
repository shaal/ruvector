# ruvector-solver-wasm

WASM bindings for the RuVector sublinear-time solver (`ruvector-solver`).
Exposes a `JsSolver` class for solving sparse linear systems, computing
Personalized PageRank, and estimating solve complexity from the browser or
any WASM runtime.

## Important files
- `Cargo.toml` - `crate-type = ["cdylib", "rlib"]`. Pulls `ruvector-solver`
  with `wasm + neumann + forward-push + cg` features. Uses
  `getrandom = "0.2"` with `js` for WASM RNG. Release: `opt-level = "s"`,
  LTO. Lints relaxed (research-tier).
- `src/lib.rs` - `init()`, `version()`, and `JsSolver` (the main JS handle).
- `src/utils.rs` - panic hook + helpers (`set_panic_hook`, `console_log`,
  `csr_from_js_arrays`).

## Public API surface
- `init()` (#[wasm_bindgen(start)]) - panic hook + load log.
- `version()` -> String.
- `JsSolver` - construct, then `solve(values, colIdx, rowPtrs, n_rows,
  n_cols, rhs)`. Accepts CSR `Float32Array` / `Uint32Array` directly from
  JS without copy.
- Exports `Algorithm`, `ComplexityClass`, `ComplexityEstimate`,
  `SparsityProfile`, `CsrMatrix` (re-exported from `ruvector-solver`).

## Tests
- `dev-dependencies = wasm-bindgen-test`. No `tests/` dir.

## Related
- `../ruvector-solver` - native Rust solver crate.
- Sibling WASM bindings: `ruvector-wasm`, `ruvector-graph-wasm`.
