# prime-radiant/src

Source root for the universal coherence engine.

## Top-level files

- `lib.rs` — crate entry with module docs and the layered architecture diagram. Re-exports engine, gate, governance, witness, policy types.
- `error.rs` — `PrimeRadiantError`, `Result` alias.
- `events.rs` — internal event bus / channel plumbing.
- `types.rs` — shared value objects (ids, energy, residual scalars).

## Subdirectories

- `attention/` — adapters into ruvector-attention (topology, MoE, PDE diffusion).
- `coherence/` — residual + energy + spectral + incremental engine.
- `cohomology/` — sheaf cohomology (H^0/H^1), coboundary, obstruction.
- `distributed/` — distributed coherence state (Raft adapter).
- `execution/` — 4-lane action ladder + gate dispatch + executor.
- `governance/` — policy, witness, lineage, repository.
- `gpu/` — wgpu acceleration + WGSL shaders.
- `hyperbolic/` — Poincare depth + energy hierarchy.
- `learned_rho/` — learned restriction map (rho) training + inference.
- `mincut/` — mincut partitioning + isolation metrics.
- `neural_gate/` — neural gate config, encoding, decision.
- `ruvllm_integration/` — bridge into ruvllm (memory, validators, pattern).
- `security/` — input limits and validation.
- `signal/` — ingestion, normalization, validation.
- `simd/` — SIMD inner loops for energy/matrix/vector ops.
- `sona_tuning/` — sona-based threshold tuner + EWC++.
- `storage/` — pluggable storage backends.
- `substrate/` — graph: nodes, edges, restriction maps, repository.
- `tiles/` — 256-tile coherence fabric (cognitum-gate-kernel bridge).

See `lib.rs` doc comment for the canonical architecture diagram and ADR-014 dependency wiring.
