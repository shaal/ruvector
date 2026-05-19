# prime-radiant/src/spectral

Spectral analysis subsystem: Lanczos eigensolver, Cheeger constant, spectral clustering, energy, and collapse.

## Files

- `mod.rs` - Module surface.
- `types.rs` - Core spectral types.
- `analyzer.rs` (~21KB) - High-level spectral analyzer.
- `lanczos.rs` - Lanczos iteration for large symmetric matrices.
- `cheeger.rs` (~18KB) - Cheeger constant / conductance.
- `clustering.rs` (~21KB) - Spectral clustering.
- `energy.rs` - Spectral energy metrics.
- `collapse.rs` (~29KB) - Spectral collapse operators.

## Related

- ADR: `../../docs/adr/ADR-004-spectral-invariants.md`.
- Bench: `../../benches/spectral_bench.rs`.
- Solver backend: `crates/ruvector-solver`.
