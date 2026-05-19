# ruvector-gnn-wasm

WebAssembly bindings for `ruvector-gnn`: GNN layer forward passes, tensor compression with adaptive level selection, differentiable search with soft attention, and hierarchical forward propagation — all callable from JS/TS.

## Important files

- `Cargo.toml` — `cdylib + rlib`. Depends on `ruvector-gnn` (wasm features), `wasm-bindgen`, `js-sys`, `serde-wasm-bindgen`. `wasm-opt = false` in release profile.
- `package.json` — npm metadata for the wasm-pack output (consumed by the JS workspace).
- `src/lib.rs` — entire wasm binding surface in a single file.

## Public API surface (`#[wasm_bindgen]`)

- `init` — installs panic hook.
- Query configuration struct for differentiable search (serde-driven).
- Wrappers around: `RuvectorLayer`, `CompressedTensor`, `CompressionLevel`, `TensorCompress`, `differentiable_search`, `hierarchical_forward`.

## Related

- `crates/ruvector-gnn` — upstream Rust crate.
- `crates/ruvector-attention-unified-wasm` — broader wasm bundle that includes GNN attention.
- `crates/ruvector-mincut-wasm` — sibling narrow wasm crate.
