# ruvector-dag/src

Source tree for the DAG / attention / SONA / mincut / healing / QuDAG stack.

## Modules (always available)

- `lib.rs` — crate root; conditional module declarations and re-exports.
- `dag/` — `QueryDag`, `OperatorNode`, traversal iterators, serialization.
- `attention/` — seven graph-aware attention mechanisms.
- `mincut/` — sub-polynomial mincut + bottleneck analysis.

## Modules behind `feature = "full"` (non-WASM)

- `sona/` — Self-Optimising Neural Architecture: MicroLoRA, EWC++, reasoning bank, trajectory.
- `healing/` — anomaly detection, drift detector, index-health checker, orchestrator, strategies.
- `qudag/` — quantum-resistant DAG protocol: client, consensus, network, proposals, sync, `crypto/`, `tokens/`.

See `../CLAUDE.md`.
