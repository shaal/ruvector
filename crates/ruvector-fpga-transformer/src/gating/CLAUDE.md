# ruvector-fpga-transformer/src/gating

Gating layer: decides whether (and how) to run inference based on coherence and policy.

## Files

- `mod.rs` — module entry + `Gate` trait.
- `coherence_gate.rs` — `DefaultCoherenceGate` integrating mincut coherence.
- `policy_gate.rs` — policy-driven gate (rules, allow/deny lists, rate limits).
