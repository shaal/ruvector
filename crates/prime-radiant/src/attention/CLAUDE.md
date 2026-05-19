# prime-radiant/src/attention

Attention-weighted residuals: applies topology-gated / MoE / PDE diffusion attention from `ruvector-attention` to weight per-edge coherence residuals.

Three attention modes: `Stable`, `Cautious`, `Freeze`.

## Files

- `mod.rs` — module wiring; re-exports `AttentionAdapter`, `AttentionCoherenceConfig`, `DiffusionSmoothing`, `SmoothedEnergy`.
- `adapter.rs` — `AttentionAdapter`: bridges substrate edges into ruvector-attention's interfaces.
- `config.rs` — `AttentionCoherenceConfig` (mode, head count, smoothing factor).
- `topology.rs` — topology-gated attention path (structural coherence as permission signal).
- `moe.rs` — MoE routing over specialized residual experts.
- `diffusion.rs` — PDE-style smoothing of energy across the substrate.

## Related

- `crates/ruvector-attention` — underlying attention implementations.
