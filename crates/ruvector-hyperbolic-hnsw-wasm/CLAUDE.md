# ruvector-hyperbolic-hnsw-wasm

WebAssembly bindings for `ruvector-hyperbolic-hnsw`: hierarchy-aware vector search in the browser using Poincare ball embeddings.

## Layout

- `Cargo.toml` — `crate-type = ["cdylib", "rlib"]`, deps: `ruvector-hyperbolic-hnsw` (default-features=false), `wasm-bindgen`, `js-sys`, `web-sys`, `getrandom` (js), `serde`, `serde-wasm-bindgen`. Optional `parallel` feature pulls `rayon` + `wasm-bindgen-rayon`. Default feature: `console_error_panic_hook`. wasm-opt = `-O3 --enable-simd`.
- `Cargo.lock` — committed (because cdylib).
- `src/lib.rs` — sole source file. Defines `HyperbolicIndex` JS class plus standalone bindings: `poincareDistance`, `mobiusAdd`, `expMap`, `logMap` (re-exporting math from the underlying crate).

## Public API surface (JS)

- `HyperbolicIndex(ef_search, curvature)` — constructor, `.insert(vec)`, `.search(query, k)`.
- Math helpers: `poincareDistance(a, b, c)`, `mobiusAdd`, `mobiusScalarMult`, `expMap`, `logMap`, `projectToBall`, `frechetMean`.

## Tests

- Dev-dep `wasm-bindgen-test`; no separate tests folder.

## Related crates

- `crates/ruvector-hyperbolic-hnsw` — native crate this wraps.
