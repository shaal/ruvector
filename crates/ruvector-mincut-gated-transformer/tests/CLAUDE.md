# ruvector-mincut-gated-transformer/tests

Integration tests (`cargo test --test <name>`).

## Files

- `integration.rs` — end-to-end smoke test.
- `determinism.rs`, `determinism_extended.rs` — verify the "same inputs → same outputs" guarantee.
- `verification.rs` — broader verification suite.
- `gate.rs` — coherence-gate behavior.
- `energy_gate.rs` — energy-based gate behavior (`energy_gate` feature).
- `mod_routing.rs` — Mixture-of-Depths routing correctness.
- `early_exit.rs` — early-exit layer-skip correctness.
- `sparse_attention.rs` — mincut-aware sparse attention.
- `spectral.rs` — spectral positional encoding.
- `spike_attention.rs` — spike-driven attention.
