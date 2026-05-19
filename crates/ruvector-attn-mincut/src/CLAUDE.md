# ruvector-attn-mincut/src

All source for the min-cut attention operator.

## Files

- `lib.rs` — crate-level doc, module declarations, public re-exports.
- `config.rs` — `MinCutConfig` (thresholds, capacities, hysteresis params, witness toggle).
- `graph.rs` — `AttentionGraph`, `Edge`, `graph_from_logits(...)` to lift QKᵀ into a weighted DAG.
- `mincut.rs` — Dinic's max-flow / min-cut algorithm with reusable BFS-level / DFS-blocking-flow helpers.
- `gating.rs` — `attn_softmax` baseline, `attn_mincut` gated operator, returns `AttentionOutput { values, gates, witness }`.
- `hysteresis.rs` — `HysteresisTracker` keeps gates temporally stable.
- `witness.rs` — SHA-256-based witness records for replay/audit.
