# ruvector-mincut-wasm

WASM bindings for `ruvector-mincut`. Exposes the dynamic min-cut, three-level hierarchy, local-k-cut, and the wrapper API to JS/TS, including the paper algorithms from arXiv:2512.13105.

## Important files

- `Cargo.toml` — `cdylib + rlib`. Depends on `ruvector-mincut` (default-features off, `wasm` feature). Uses `wasm-bindgen`, `wasm-bindgen-futures`, `js-sys`, `serde-wasm-bindgen`, `console_error_panic_hook`. `wasm-opt = false`.
- `src/lib.rs` — entire `#[wasm_bindgen]` surface in one file.

## Public API surface (`#[wasm_bindgen]`)

- `WasmMinCut` — basic dynamic min-cut (insert/delete/query).
- `WasmThreeLevelHierarchy` — 3-level decomposition (Expander → Precluster → Cluster).
- `WasmLocalKCut` — deterministic local k-cut with 4-color coding.
- `WasmMinCutWrapper` — full API with connectivity-curve analysis.

Each exposes `fromEdges` / `insertEdge` / `deleteEdge` / `minCutValue` / `stats` style methods (see `src/lib.rs` and `crates/ruvector-mincut`).

## Related

- `crates/ruvector-mincut` — upstream Rust crate (algorithms, docs, ADRs).
- `crates/ruvector-attention-unified-wasm`, `crates/ruvector-gnn-wasm` — sibling wasm crates.
