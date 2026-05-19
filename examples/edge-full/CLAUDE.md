# examples/edge-full

Pre-built distribution of the `@ruvector/edge-full` WASM toolkit: bundles every RuVector WASM module (vector search, graph DB, neural networks, DAG workflows, SQL/SPARQL/Cypher, ONNX inference) for in-browser / edge use.

## Layout
- `pkg/` - The publishable npm package. Contains the unified entrypoint plus one subdirectory per WASM module.

## Tech stack
- Rust crates compiled to WASM via `wasm-bindgen`.
- ES modules (`type: module`), MIT licensed, no runtime dependencies.

## Related
- See repo crates such as `crates/ruvector-edge`, `crates/ruvector-graph`, `crates/ruvector-dag`, `crates/ruvector-sona`, `crates/rvlite`, `crates/ruvector-onnx-embeddings` (the sources for these WASM modules).
