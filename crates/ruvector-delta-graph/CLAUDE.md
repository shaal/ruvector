# ruvector-delta-graph

Delta operations for graph structures: incremental edge / node updates with delta-aware traversal. Builds on `ruvector-delta-core`'s generic delta abstractions for streaming graph mutations.

## Important files

- `Cargo.toml` — deps: `ruvector-delta-core`, `parking_lot`, `dashmap`, `smallvec`, `thiserror`. Optional `serde` (feature), optional `rayon` (`parallel` feature).
- `src/lib.rs` — doc, module decls, re-exports of `EdgeDelta`, `EdgeOp`, `NodeDelta`, `PropertyDelta`, `DeltaAwareTraversal`, `TraversalMode`, `GraphDeltaError`.
- `src/edge_delta.rs` — `EdgeDelta`, `EdgeOp` (add, remove, update weight) and batch application.
- `src/node_delta.rs` — `NodeDelta`, `PropertyDelta` (scalar + vector property changes).
- `src/traversal.rs` — `DeltaAwareTraversal`, `TraversalMode`: walk the graph honoring pending deltas.
- `src/error.rs` — crate error enum (`GraphDeltaError`, `Result`).

## Public API surface

`EdgeDelta`, `EdgeOp`, `NodeDelta`, `PropertyDelta`, `DeltaAwareTraversal`, `TraversalMode`, plus the re-exported `Delta`, `DeltaStream`, `VectorDelta` traits from `ruvector-delta-core`.

## Features

`parallel` enables `rayon`; `serde` enables `serde`/`serde_json`.

## Related

- `crates/ruvector-delta-core` — generic delta primitives.
- `crates/ruvector-gnn`, `crates/ruvector-graph` — graph data structures that may consume these deltas.
