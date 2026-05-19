# 08-meta-simulation-consciousness

Standalone research crate: O(N^3) integrated information (Phi) for
ergodic systems via meta-simulation. Claims to reduce IIT Phi from
exponential to polynomial time on ergodic dynamics.

## Files

- `Cargo.toml` — standalone `[workspace]`; package
  `meta-sim-consciousness`. Runtime dep `rayon`. Dev-dep `criterion`.
  Bench `meta_sim_benchmarks`.
- `RESEARCH.md`, `BREAKTHROUGH_HYPOTHESIS.md`, `RESEARCH_SUMMARY.md`,
  `INDEX.md`, `complexity_analysis.md` — theory.
- `Cargo.lock` — pinned.
- `src/lib.rs` — re-exports.
- `src/closed_form_phi.rs` — closed-form Phi for tractable subsystems.
- `src/ergodic_consciousness.rs` — ergodic-system Phi pipeline.
- `src/hierarchical_phi.rs` — hierarchical Phi computation.
- `src/meta_sim_awareness.rs` — meta-simulation awareness layer.
- `src/simd_ops.rs` — SIMD kernels.
- `benches/meta_sim_benchmarks.rs` — Criterion suite.

## Build / Bench

```bash
cd examples/exo-ai-2025/research/08-meta-simulation-consciousness
cargo build --release
cargo bench
```

## Related

- `../../../ecosystem-consciousness/` — small-scale Phi demo
- `../06-federated-collective-phi/` — distributed Phi
