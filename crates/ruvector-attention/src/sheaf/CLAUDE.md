# ruvector-attention/src/sheaf

Sheaf attention (Coherence-Gated Transformer per ADR-015). Behind the `sheaf` feature.

## Files

- `mod.rs` — module entry.
- `attention.rs` — sheaf-attention forward (uses restriction maps on edges).
- `restriction.rs` — restriction-map abstraction.
- `router.rs` — sparse routing through the sheaf.
- `sparse.rs` — sparse sheaf-attention kernel.
- `early_exit.rs` — layer-level early-exit when coherence is satisfied.

## Related

- `crates/prime-radiant/src/cohomology/` and `src/coherence/` consume sheaf attention through their adapter.
