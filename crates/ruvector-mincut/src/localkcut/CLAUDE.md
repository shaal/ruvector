# ruvector-mincut/src/localkcut

Deterministic local k-cut discovery (paper-faithful 4-color coding scheme). Exposed via WASM as `WasmLocalKCut`.

## Files

- `mod.rs` — local-k-cut façade.
- `deterministic.rs` — deterministic 4-color-coding implementation.
- `paper_impl.rs` — paper-faithful reference implementation (closer to arXiv pseudocode).

See `examples/localkcut_demo.rs` and tests `tests/localkcut_integration.rs`, `tests/localkcut_paper_integration.rs`.
