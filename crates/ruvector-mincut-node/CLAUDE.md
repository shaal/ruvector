# ruvector-mincut-node

Node.js bindings (NAPI-RS) for `ruvector-mincut`. Exposes subpolynomial-time
dynamic minimum cut (arXiv:2512.13105), the 3-level hierarchical decomposition,
deterministic local k-cut with 4-color coding, and connectivity-curve analysis to
Node.

## Layout

- `Cargo.toml` — `crate-type = ["cdylib"]`. Depends on `ruvector-mincut` (feature
  `monitoring`), napi, napi-derive.
- `build.rs` — calls `napi-build`.
- `src/lib.rs` — `#[napi]`-exported wrappers (`JsEdge`, `JsStats`, `MinCut`,
  `ThreeLevelHierarchy`, `LocalKCut`, `MinCutWrapper`).

## Exposed JS API

- `MinCut` — basic dynamic min-cut (insert/delete/query).
- `ThreeLevelHierarchy` — Expander -> Precluster -> Cluster decomposition.
- `LocalKCut` — deterministic local k-cut with 4-color coding.
- `MinCutWrapper` — full API including connectivity-curve analysis.

## Related

- `crates/ruvector-mincut` — underlying native crate.
