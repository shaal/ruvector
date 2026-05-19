# ruvector-attn-mincut

Min-cut gating attention operator: a graph-based alternative to softmax attention. Builds a weighted directed graph from QKᵀ logits, computes a minimum cut via Dinic's max-flow, and gates irrelevant edges before row-softmax / value multiplication. Pure-Rust, no SIMD deps.

## Important files

- `Cargo.toml` — `rlib` only. Deps: `serde`, `serde_json`, `sha2`. No `ruvector-mincut` dep — this is a self-contained operator built from scratch (distinct from the broader `ruvector-mincut` crate).
- `src/lib.rs` — doc, module declarations, re-exports.
- `src/config.rs` — `MinCutConfig` with sane defaults.
- `src/graph.rs` — `graph_from_logits`, `AttentionGraph`, `Edge`: build the flow graph from attention logits.
- `src/mincut.rs` — Dinic's max-flow / min-cut solver.
- `src/gating.rs` — `attn_softmax` (baseline) and `attn_mincut` (gated) operators; `AttentionOutput` struct.
- `src/hysteresis.rs` — `HysteresisTracker` smooths gate decisions over time.
- `src/witness.rs` — SHA-256 witness logging for determinism / audit.

## Public API surface

`MinCutConfig`, `attn_softmax`, `attn_mincut`, `AttentionOutput`, `graph_from_logits`, `AttentionGraph`, `Edge`, `HysteresisTracker`, witness helpers.

## Related

- `crates/ruvector-mincut-gated-transformer` — full transformer that uses min-cut gating.
- `crates/ruvector-mincut` — generic dynamic min-cut algorithms (much larger, different focus).
- `crates/ruvector-attention` — baseline softmax attention library.
