# 10-thermodynamic-learning

Standalone research crate: physics-based learning approaching Landauer
limits. Includes Friston free-energy agents, equilibrium propagation,
Landauer-bounded learning, and reversible neural nets.

## Files

- `Cargo.toml` — standalone `[workspace]`; package
  `thermodynamic-learning`. Runtime dep `rand`; dev-dep `criterion`.
  Bench `thermodynamic_bench`.
- `RESEARCH.md`, `BREAKTHROUGH_HYPOTHESIS.md`, `SUMMARY.txt`,
  `physics_foundations.md` — theory.
- `Cargo.lock` — pinned.
- `src/lib.rs` — re-exports.
- `src/free_energy_agent.rs` — Friston free-energy agent.
- `src/equilibrium_propagation.rs` — equilibrium propagation learning.
- `src/landauer_learning.rs` — Landauer-bounded learning rule.
- `src/reversible_neural.rs` — reversible neural net layers.
- `src/novel_algorithms.rs` — novel thermodynamic learning algorithms.
- `src/simd_ops.rs` — SIMD kernels.
- `benches/thermodynamic_bench.rs` — Criterion suite.

## Build / Bench

```bash
cd examples/exo-ai-2025/research/10-thermodynamic-learning
cargo build --release
cargo bench
```

## Related

- `../../crates/exo-core/src/thermodynamics.rs`
- `../../crates/exo-exotic/src/thermodynamics.rs`
- `../../../../crates/thermorust/`
