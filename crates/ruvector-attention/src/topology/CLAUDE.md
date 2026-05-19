# ruvector-attention/src/topology

Topology-gated attention: use structural coherence as a permission signal that gates whether attention fires for a given query.

## Files

- `mod.rs` — module entry.
- `gated_attention.rs` — main gated attention kernel.
- `coherence.rs` — coherence-score computation feeding the gate.
- `policy.rs` — gating policy (thresholds, modes).
