# ruvector-delta-graph/src

All source for graph deltas.

## Files

- `lib.rs` — crate doc, module decls, public re-exports.
- `edge_delta.rs` — `EdgeDelta` and `EdgeOp { Add, Remove, UpdateWeight }`; batch application semantics.
- `node_delta.rs` — `NodeDelta` and `PropertyDelta` (including vector-property deltas via `ruvector_delta_core::VectorDelta`).
- `traversal.rs` — `DeltaAwareTraversal` walks graph honoring not-yet-applied deltas; `TraversalMode` selects strategy.
- `error.rs` — `GraphDeltaError`, `Result` typedef.
