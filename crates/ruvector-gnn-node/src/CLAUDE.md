# ruvector-gnn-node/src

Single-file NAPI glue: exports Rust `ruvector-gnn` types into Node.js as
`#[napi]` classes and functions.

## Files
- `lib.rs` - `RuvectorLayer` class binding, `CompressedTensor` /
  `CompressionLevel` / `TensorCompress` bindings, and `differentiable_search`
  / `hierarchical_forward` free functions.
