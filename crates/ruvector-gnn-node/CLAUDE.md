# ruvector-gnn-node

Node.js bindings for `ruvector-gnn` via NAPI-RS. Ships a native addon plus a
JS package façade (`package.json`). Exposes graph neural network layers,
tensor compression, and differentiable search to Node.

## Important files
- `Cargo.toml` - `crate-type = ["cdylib"]`. Workspace `napi` + `napi-derive`.
  Release: `lto = true`, `strip = true`. Many lints relaxed (research-tier).
- `build.rs` - one-liner invoking `napi-build`.
- `package.json` - npm metadata; selects the right prebuilt `.node` based on
  platform via the `npm/*` sub-packages.
- `.npmignore` - excludes Rust sources from the published npm tarball.
- `src/lib.rs` - all `#[napi]` bindings (single file).
- `npm/<platform>/` - per-target packages each containing a prebuilt `.node`
  binary. Currently macOS arm64 has a prebuilt; others are placeholders.
- `examples/basic.js` - hello-world Node example.
- `test/basic.test.js` - basic Node test.
- `.github/` - workflows for prebuild + publish.

## Public API surface
- `RuvectorLayer` (`#[napi]`) - GNN layer for HNSW topology; constructor
  takes `input_dim`, `hidden_dim`, `heads`, `dropout`.
- `CompressedTensor`, `CompressionLevel`, `TensorCompress` - tensor
  compression bindings.
- Free functions: `differentiable_search(...)`,
  `hierarchical_forward(...)`.

## Related
- `../ruvector-gnn` - the underlying Rust library.
- Sibling NAPI binding: `ruvector-graph-transformer-node`.
